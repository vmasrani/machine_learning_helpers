from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
import time
from string import Formatter


def timeit(message=None):
    def decorator(method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if message:
                print(f'{message}: {te - ts:.2f}s')
            else:
                print(f'{method.__name__}: {te - ts:.2f}s')
            return result
        return timed
    return decorator


@dataclass
class GPTAgent:
    name: str
    prompt: str
    instructions: str
    model: str
    timeout: int
    tools: List[str]
    output_pattern: str

    def _get_tools(self) -> List[Dict[str, str]]:
        """Convert tool names to OpenAI tool format"""
        tool_mapping = {
            "code_interpreter": {"type": "code_interpreter"},
            "retrieval": {"type": "retrieval"},
            "function": {"type": "function"}
        }
        return [tool_mapping[tool] for tool in self.tools]

    def _create_assistant(self) -> str:
        print("Creating new assistant...")
        assistant = self.client.beta.assistants.create(
            name=self.name,
            instructions=self.instructions,
            model=self.model,
            tools=self._get_tools()
        )
        return assistant.id

    def __init_gpt(self):
        """Initialize OpenAI client and assistant"""
        self.client = OpenAI()
        self.assistant_id = self._create_assistant()

    @timeit(message="Done")
    def upload_file(self, file_path: Path) -> str:
        print(f"Uploading file: {file_path}")
        response = self.client.files.create(
            file=open(file_path, "rb"),
            purpose="assistants"
        )
        return response.id

    @timeit(message="Saving output file")
    def save_output_file(self, file_id: str, original_path: Path) -> None:
        print("Saving output file...")
        new_filename = original_path.parent / self.output_pattern.format(
            stem=original_path.stem,
            suffix=original_path.suffix
        )
        content = self.client.files.content(file_id)
        with open(new_filename, "wb") as f:
            f.write(content.read())

    def process_response(self, response, file_path: Path) -> None:
        cleaned_file_id = response.attachments[0].file_id if response.attachments else None
        if cleaned_file_id:
            self.save_output_file(cleaned_file_id, file_path)
        else:
            print("No output file was generated.")
            print("Assistant's response:")
            for content_block in response.content:
                if content_block.type == 'text':
                    print(content_block.text.value)

    def _get_required_kwargs(self) -> set:
        """Extract required template variables from prompt"""
        return {
            fname for _, fname, _, _
            in Formatter().parse(self.prompt)
            if fname is not None
        }

    def _validate_and_prepare_kwargs(self, file_path, **kwargs) -> tuple[Path, dict]:
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        all_kwargs = {
            'file_path': str(file_path),
            **kwargs
        }

        required_kwargs = self._get_required_kwargs()
        if missing_kwargs := required_kwargs - set(all_kwargs.keys()):
            raise ValueError(f"Missing required template variables: {missing_kwargs}")

        return file_path, all_kwargs

    @timeit(message="Creating thread and starting run")
    def _create_thread(self, formatted_prompt: str, file_id: str) -> tuple[str, str]:
        thread = self.client.beta.threads.create(
            messages=[{
                "role": "user",
                "content": formatted_prompt,
                "attachments": [{
                    "file_id": file_id,
                    "tools": self._get_tools()
                }]
            }]
        )

        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant_id
        )

        return thread.id, run.id

    @timeit(message="Waiting for completion")
    def _wait_for_completion(self, thread_id: str, run_id: str) -> None:
        start_time = time.time()
        print("Waiting for assistant to complete...")
        while True:
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Assistant did not complete within {self.timeout} seconds")

            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            if run.status == 'completed':
                break
            time.sleep(2)
            print(".", end="", flush=True)

    @timeit(message="Total GPT processing time")
    def run(self, file_path, **kwargs) -> None:
        """
        Run the GPT agent on a file with required template variables.
        The file_path is automatically available as {file_path} in the prompt.

        Args:
            file_path: Path to the file to process
            **kwargs: Additional template variables required by the prompt
        """
        # Validate and prepare inputs
        file_path, all_kwargs = self._validate_and_prepare_kwargs(file_path, **kwargs)

        # Initialize GPT if needed
        if not hasattr(self, 'client'):
            self.__init_gpt()

        # Upload file and format prompt
        file_id = self.upload_file(file_path)
        formatted_prompt = self.prompt.format(**all_kwargs)

        # Create thread and start run
        thread_id, run_id = self._create_thread(formatted_prompt, file_id)

        # Wait for completion
        self._wait_for_completion(thread_id, run_id)

        # Process results
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        self.process_response(messages.data[0], file_path)
