from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------------
MODEL_NAME = "huggyllama/llama-7b"
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 100  # small, enough for one question
FEEDBACK = True

# -----------------------------
print("Loading model... this may take a while!")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)
print("Model loaded. Starting interview...\n")

# -----------------------------
chat_history = (
    "You are an experienced IT interviewer.\n"
    "Rules:\n"
    "- Ask **one question at a time**.\n"
    "- Do not include introductions or pre-filled dialogue.\n"
    "- Wait for the candidate's answer before asking the next question.\n"
    "- Optionally give short feedback after each answer.\n"
    "Start the interview with the first IT-related question.\nInterviewer: "
)


# -----------------------------
while True:
    # Generate one question
    inputs = tokenizer(chat_history, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.eos_token_id
    )

    question = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"\nInterviewer: {question}")

    # Candidate answer
    answer = input("You: ")
    if answer.lower() == "exit":
        print("\nInterview ended. Goodbye!")
        break

    # Update chat history for next round
    if FEEDBACK:
        chat_history += f"{question}\nCandidate: {answer}\nInterviewer: Provide a short feedback on the answer and ask the next question.\n"
    else:
        chat_history += f"{question}\nCandidate: {answer}\nInterviewer: Ask the next question.\n"
