
from ..abstract_runner import AbstractRunner


class FalconInstructRunner(AbstractRunner):


    @staticmethod
    def start_prompt():
        return """
                Below is an instruction that describes a task. Write a response that appropriately completes the request.
                ### Instruction:
                The following text is an extract from a wikipedia page. The text is about a person but the person is referred to as <mask>.
                Please give the name of the person referred to as <mask> and only the name. If you don't know the name,
                give your best guess. Do not include any other information in your response.

                The text:

                """

    @staticmethod
    def end_prompt():
        return """

                ### Response:
                """

    @staticmethod
    def names():
        return {
            "falcon_instruct-40b": "tiiuae/falcon-40b-instruct",
            "falcon_instruct-7b": "tiiuae/falcon-7b-instruct",
        }
    
    @staticmethod
    def sizes():
        return {
            "XL": "falcon_instruct-7b",
            "XXL": "falcon_instruct-40b",
        }
        
    @staticmethod
    def batch_sizes():
        return {
            "falcon_instruct-7b": 8,
            "falcon_instruct-40b": 2,
        }