from openai import OpenAI


class OpenAIClient:
    def __init__(self, base_url=None):
        if base_url:
            self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        else:
            self.client = OpenAI()

    def get_text(
        self,
        text,
        model,
        max_tokens=500,
        temperature=0.0,
        top_p=1.00,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        system_message=None,
    ):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": text})

        # Try making the API call
        try:
            response = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                messages=messages,
            )
        except Exception as e:
            raise Exception(f"Failed to create completion with OpenAI API: {str(e)}")

        # Check if the response has valid data
        if response.choices and len(response.choices) > 0:
            first_choice = response.choices[0]

            if first_choice.message and first_choice.message.content:
                return str(first_choice.message.content)
            else:
                raise Exception(
                    "Response from OpenAI API does not "
                    "contain 'message' or 'content'."
                )
        else:
            raise Exception(
                "Response from OpenAI API does not contain "
                "'choices' or choices list is empty."
            )
