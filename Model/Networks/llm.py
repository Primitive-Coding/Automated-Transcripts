from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration


available_models = ["gpt-neo", "t5-small"]


class LocalLLM:
    def __init__(self, model_name="gpt-neo") -> None:
        # Load the tokenizer and model
        self.model_name = model_name.lower()
        if self.model_name == "gpt-neo":
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        elif self.model_name == "t5-small":
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def get_output(
        self, text: str, max_len: int = 50, q_and_a: bool = False, multi: bool = False
    ):

        if q_and_a:
            question = text
            text = "question: " + question + " context: "
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_len)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = output.split("\n")
        output = [s for s in output if s]  # Remove '' from list.

        if self.model_name == "gpt-neo":
            if multi:
                return output[1:]
            else:
                return output[1]
        else:
            return output[0]

    def display_q_and_a(
        self, text: str, clean_input: bool = False, answer_len: int = 50
    ):
        if clean_input:
            text = self._clean_input_text(text)

        if self.model_name == "gpt-neo":
            q_and_a = False
        else:
            q_and_a = True
        answer = self.get_output(text, max_len=answer_len, q_and_a=q_and_a)

        print(
            f"""
=========================
Question: {text}
              
Answer: {answer}
              """
        )

    def _clean_input_text(self, text):

        if text[-1] == ".":
            text[-1] = "?"

        if text[-1] != "?":
            text += "?"

        return text
