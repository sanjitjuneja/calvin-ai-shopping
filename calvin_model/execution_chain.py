from typing import List

from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import Field

class ExecutionChain(LLMChain):
    """Chain to execute tasks."""

    vectorstore: VectorStore = Field(init=False)

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = True
    ) -> LLMChain:
        """Get the response parser."""
        execution_template = (
            "You are Calvin, an AI Intelligent Shopper who who enhances people's buying experience when shopping online and performs one task based on the following objective: {objective}."
            " Take into account these previously completed tasks: {context}."
            " Your task: {task}."
            " Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose, vectorstore=vectorstore)

    def _get_top_tasks(self, query: str, k: int) -> List[str]:
        """Get the top k tasks based on the query."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return []
        sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
        return [str(item.metadata["task"]) for item in sorted_results]

    def execute_task(self, objective: str, task: str, k: int = 5) -> str:
        """Execute a task."""
        context = self._get_top_tasks(query=objective, k=k)
        return self.run(objective=objective, context=context, task=task)