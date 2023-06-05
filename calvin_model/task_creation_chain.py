from typing import Dict, List
from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM

class TaskCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            partial_variables={"objective": objective},
            input_variables=["result", "task_description", "incomplete_tasks"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def get_next_task(
        self, result: Dict, task_description: str, task_list: List[str]
    ) -> List[Dict]:
        """Get the next task."""
        incomplete_tasks = ", ".join(task_list)
        response = self.run(
            result=result,
            task_description=task_description,
            incomplete_tasks=incomplete_tasks,
        )
        new_tasks = response.split("\n")
        return [
            {"task_name": task_name} for task_name in new_tasks if task_name.strip()
        ]