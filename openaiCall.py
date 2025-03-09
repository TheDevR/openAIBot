from openai import OpenAI  # This lets us use OpenAI's tools in our code.
import json  # Helps us work with JSON, a way computers can exchange data.

class openaiBot:
    # Keeps track of how many words or 'tokens' we've used in both questions and answers.
    cumulative_tokens = {'prompt':0, 'completion':0}
    latest_tokens = {'prompt':0, 'completion':0}
    
    def __init__(self, api_key, job="You are helpful", model="gpt-3.5-turbo-0125", stateless=True, jsonMode=False):
        self.client = OpenAI(api_key=api_key)  # This line logs us into OpenAI's system.
        self.model = model  # This decides which AI brain to talk to.
        self.job = job  # This is like giving the AI a first instruction or purpose.
        self.messages = [{"role": "system", "content": self.job}]  # This is where we start keeping track of our conversation.
        self.stateless = stateless  # This decides if the AI should 'remember' past messages within a chat.
        self.jsonMode = jsonMode  # This decides how we want the answers back: as structured data or plain text.

    def reset_chat(self):
        # This function clears the chat, starting over with just the 'job' message.
        self.messages = [{"role": "system", "content": self.job}]

    def chat(self, user_message, temperature=1, max_tokens=1000, frequency_penalty=0, presence_penalty=0):
        # Here we send a message to the AI and get back its response.
        # 'temperature' affects creativity: lower for more predictable, higher for more varied responses.
        # 'max_tokens' limits how long the answer can be.
        # 'frequency_penalty' makes repeating the same thing less likely.
        # 'presence_penalty' makes new ideas more likely.
        self.add_message("user", user_message)  # We record the message we're sending in our chat history.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format={"type": "json_object" if self.jsonMode else "text"})
        
        # After getting the response, we update our record of how many words we've used.
        self.cumulative_tokens['prompt'] += response.usage.prompt_tokens
        self.cumulative_tokens['completion'] += response.usage.completion_tokens
        self.latest_tokens['prompt'] = response.usage.prompt_tokens
        self.latest_tokens['completion'] = response.usage.completion_tokens
        
        assistant_message = response.choices[0].message.content  # This pulls out the AI's answer from the response.
        if not self.stateless:
            self.add_message("assistant", assistant_message)  # If we're keeping track of the conversation, record this answer.
        else:
            self.messages.pop()  # If not, forget the last thing we said.
        return json.loads(assistant_message) if self.jsonMode else assistant_message  # Returns the answer in the format we want.
    
    def show_tokens(self):
        # Shows us how many words we've used in total, helping manage costs.
        latest_sum = self.latest_tokens['prompt'] + self.latest_tokens['completion']
        cumulative_sum = self.cumulative_tokens['prompt'] + self.cumulative_tokens['completion']
        print("Token Usage Details:")
        print("Latest and cumulative token counts here.")
        
    def add_message(self, role, content):
        # Adds a new part to the conversation. 'role' is who's talking, and 'content' is what they said.
        self.messages.append({"role": role, "content": content})

    def get_chat_history(self):
        # Gives us the whole conversation back so we can see how we got here.
        return self.messages
