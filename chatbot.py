import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Streamlit page setup
st.set_page_config(page_title="Offline Chatbot", page_icon="🤖")
st.title("💬 Offline Chatbot (DialoGPT-small)")

# Load model & tokenizer once (cached)
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Initialize chat history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Chat input
user_input = st.chat_input("Say something...")

if user_input:
    st.session_state["history"].append({"role": "user", "content": user_input})

    # Prepare model input with past chat context
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Combine all previous inputs if available
    bot_input_ids = new_user_input_ids
    if "chat_history_ids" in st.session_state:
        bot_input_ids = torch.cat([st.session_state["chat_history_ids"], new_user_input_ids], dim=-1)

    # Generate model response
    output_ids = model.generate(
        bot_input_ids,
        max_length=500,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    # Save chat history (so context is remembered)
    st.session_state["chat_history_ids"] = output_ids

    # Decode response
    bot_reply = tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.session_state["history"].append({"role": "assistant", "content": bot_reply})

# Display conversation
for chat in st.session_state["history"]:
    if chat["role"] == "user":
        st.chat_message("user").markdown(chat["content"])
    else:
        st.chat_message("assistant").markdown(chat["content"])
