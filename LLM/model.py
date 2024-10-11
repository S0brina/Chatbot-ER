import logging
import os
import json
import re
from transformers import (AutoTokenizer, AutoModelForCausalLM, pipeline)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    allowed_chars = re.compile(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ .,!?;:\'\"()-]')
    text = allowed_chars.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    cleaned_text = text.lower()
    
    return cleaned_text

def load_config(config_file):
    with open(config_file) as f:
        config_data = json.load(f)
    return config_data

def load_model_and_tokenizer(model_name, hf_token, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, use_auth_token=hf_token)
    return model, tokenizer

def set_pipeline(model_name, hf_token, pipeline_type, device="cpu"):
    model, tokenizer = load_model_and_tokenizer(model_name, hf_token, device)
    text_generator = pipeline(
        pipeline_type, model=model, tokenizer=tokenizer
    )
    return text_generator

def first_prompt(user_input, prompt_template=None):
    usr_input = clean_text(user_input)
    
    if prompt_template:
        prompt = prompt_template.format(user_input=usr_input)
    else:
        prompt = (
            f"Con base en la siguiente descripción: {usr_input}\n"
            f"Identifica lo siguiente:\n"
            f"1. Los procesos que se mencionan en la descripción.\n"
            f"2. Los roles que participan en cada proceso.\n"
            f"3. Las actividades específicas que desempeña cada rol.\n"
            f"4. Envía los resultados únicamente en el formato: <Proceso>, <Rol>, <Actividad>.\n"
            f"5. Elimina las redundancias.\n"
        )
    return prompt

def second_prompt(structured_data,prompt_template=None):
    if prompt_template:
        prompt = prompt_template.format(user_input=structured_data)
    else:
        prompt = (
            f"Con base en la siguiente información estructurada: \n\n{structured_data}\n"
            f"1. Ten en cuenta que este texto tiene el formato: <Proceso>, <Rol>, <Actividad>.\n"
            f"2. Propone un software para automatizar las actividades identificadas.\n"
            f"3. Detalla minuciosamente como será el nuevo roadmap del usuario en el software propuesto por cada actividad identificada.\n"
            f"4. Transforma cada uno de los pasos de los roadmaps a historias de usuario en formato Connextra.\n"
            f"5. Agrupa estas historias de usuario por epicas.\n"
        )
    return prompt

def generate_response(generator, prompt, ptemperature, ptop_k, ptop_p, pmax_new_tokens, plength_penalty):
    inputs = generator.tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    outputs = generator.model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask,
        num_return_sequences = 1,
        temperature = ptemperature,
        top_k = ptop_k,
        top_p = ptop_p,
        max_new_tokens = pmax_new_tokens,
        length_penalty = plength_penalty
    )

    decoded_output = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

def chat_llm(
    generator,
    prompt_template=None,
    process_temperature = 0.55, process_top_k = 40, process_top_p = 0.8, process_length_penalty = 1, process_max_new_tokens = 600,
    proposal_temperature = 0.8, proposal_top_k = 60, proposal_top_p = 0.9, proposal_length_penalty = 1.1, proposal_max_new_tokens = 2000
):
    print("------- Ejecutando chatbot, presionar enter para terminar sesión. --------")

    user_input = input("Usuario: ")

    prompt = first_prompt(user_input, prompt_template)
    structured_data = generate_response(
        generator, prompt, process_temperature, process_top_k, process_top_p, process_max_new_tokens, process_length_penalty
    )
    print("Análisis estructurado:")
    print(structured_data)

    software_prompt = second_prompt(structured_data, prompt_template)
    software_proposal = generate_response(
        generator, software_prompt, proposal_temperature, proposal_top_k, proposal_top_p, proposal_max_new_tokens, proposal_length_penalty
    )
    print("Historias de usuario generadas:")
    print(software_proposal)

def main(config_file, model_name, pipeline_type="text-generation", device="cpu", prompt_template=None):
    config_data = load_config(config_file)
    hf_token = config_data["HF_TOKEN"]

    generator = set_pipeline(model_name, hf_token, pipeline_type, device)

    # Hiperparámetros específicos para generar el proceso (generate_process)
    process_temperature = 0.55
    process_top_k = 40
    process_top_p = 0.8
    process_max_new_tokens = 600
    process_length_penalty = 1

    # Hiperparámetros específicos para generar la propuesta de software (generate_proposal)
    proposal_temperature = 0.8
    proposal_top_k = 60
    proposal_top_p = 0.9
    proposal_max_new_tokens = 2000
    proposal_length_penalty = 1.1

    chat_llm(
        generator, 
        prompt_template, 
        process_temperature, process_top_k, process_top_p,process_length_penalty,process_max_new_tokens,
        proposal_temperature, proposal_top_k, proposal_top_p,proposal_length_penalty, proposal_max_new_tokens
    )

if __name__ == "__main__": 
    main(
        config_file="config.json", 
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", 
        pipeline_type="text-generation", 
        device="cpu"
    )
