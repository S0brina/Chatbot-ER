import logging
import os
import json
import re
from transformers import (AutoTokenizer, AutoModelForCausalLM, pipeline)
from datetime import datetime

# Configuración del directorio y nombre de log con formato LOG_DATETIME
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_filename = f"LOG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=os.path.join(log_directory, log_filename),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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

def set_pipeline(model_name, hf_token, pipeline_type="text-generation", max_new_tokens=600, device="cpu"):
    model, tokenizer = load_model_and_tokenizer(model_name, hf_token, device)
    text_generator = pipeline(
        pipeline_type, model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens
    )
    return text_generator

def first_prompt(user_input, prompt_template=None):
    if prompt_template:
        usr_input = clean_text(user_input)
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
            f"4. Solo envía los roadmaps propuestos.\n"
        )
        logging.info(f"Prompt generado para propuesta de software: {prompt}")
    return prompt

def third_prompt(user_input, prompt_template=None):
    if prompt_template:
        prompt = prompt_template.format(user_input=user_input)
    else:
        prompt = (
            f"En base a los siguientes roadmaps de usuarios: {user_input}\n"
            f"1. Identifica los requerimientos funcionales del software.\n"
            f"2. Identifica los requerimientos no funcionales del software.\n"
            f"3. Transforma ambos tipos de requerimientos a historias de usuario.\n"
            f"4. Envía las historias de usuario en formato Connextra.\n"
        )
        logging.info(f"Prompt generado para historias de usuario: {prompt}")
    return prompt

def generate_process(generator, prompt, max_length=1500, temperature=0.55, top_k=40, top_p=0.8, max_new_tokens = 500):
    inputs = generator.tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    outputs = generator.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature, 
        top_k=top_k,  
        top_p=top_p,
        max_new_tokens = max_new_tokens
    )

    # Decodificar la respuesta generada
    decoded_output = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Filtrar solo las líneas que siguen el formato deseado
    pattern = r"^.*?, .*?, .*?$"
    filtered_output = "\n".join(re.findall(pattern, decoded_output, re.MULTILINE))

    # Eliminar repeticiones
    lines_seen = set()
    unique_lines = []
    for line in filtered_output.splitlines():
        if line not in lines_seen:
            unique_lines.append(line)
            lines_seen.add(line)

    # Unir las líneas únicas en el resultado final
    final_output = "\n".join(unique_lines)

    logging.info(f"Temperature: {temperature}, top_k: {top_k}, top_p: {top_p}")
    logging.info(f"Response filtrada: {filtered_output}")
    
    return final_output

def generate_proposal(generator, prompt, max_length=2000, temperature=0.8, top_k=60, top_p=0.9, max_new_tokens = 1000 ):

    inputs = generator.tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    outputs = generator.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens = max_new_tokens,
        length_penalty=1.1 
    )

    # Decodificar la respuesta
    software_proposal = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)

    logging.info(f"Temperature: {temperature}, top_k: {top_k}, top_p: {top_p}")

    logging.info(f"Propuesta de software generada: {software_proposal}")
    
    return software_proposal

def generate_user_stories(generator, prompt, max_length=2000, temperature=0.65, top_k=50, top_p=0.9, max_new_tokens = 600):
    inputs = generator.tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    outputs = generator.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens = max_new_tokens
    )

    # Decodificar la respuesta
    user_stories = generator.tokenizer.decode(outputs[0], skip_special_tokens=True)

    logging.info(f"Historias de usuario generadas: {user_stories}")
    
    return user_stories

def chat_llm(
    generator, 
    prompt_template=None, 
    process_max_length=1500, process_temperature=0.55, process_top_k=40, process_top_p=0.8, process_max_new_tokens=500,
    proposal_max_length=2000, proposal_temperature=0.8, proposal_top_k=60, proposal_top_p=0.9, proposal_max_new_tokens=1000,
    user_story_max_length=2000, user_story_temperature=0.65, user_story_top_k=50, user_story_top_p=0.9, user_stories_max_new_tokens=600
):
    print("------- Ejecutando chatbot, presionar enter para terminar sesión. --------")
  
    while True:
        user_input = input("Usuario: ")
        if not user_input:
            break
        logging.info(f"Input del usuario: {user_input}")
        
        prompt = first_prompt(user_input, prompt_template)
        structured_data = generate_process(
            generator, prompt, process_max_length, process_temperature, process_top_k, process_top_p,process_max_new_tokens
        )
        print("Análisis estructurado:")
        print(structured_data)
        
        software_prompt = second_prompt(structured_data, prompt_template)
        software_proposal = generate_proposal(
            generator, software_prompt, proposal_max_length, proposal_temperature, proposal_top_k, proposal_top_p,proposal_max_new_tokens
        )
        print("Propuesta de software:")
        print(software_proposal)
        
        user_story_prompt = third_prompt(software_proposal, prompt_template)
        user_stories = generate_user_stories(
            generator, user_story_prompt, user_story_max_length, user_story_temperature, user_story_top_k, user_story_top_p,user_stories_max_new_tokens
        )
        print("Historias de usuario generadas:")
        print(user_stories)

def main(config_file, model_name, pipeline_type="text-generation", device="cpu", prompt_template=None):
    config_data = load_config(config_file)
    hf_token = config_data["HF_TOKEN"]

    generator = set_pipeline(model_name, hf_token, pipeline_type, device)

    # Hiperparámetros específicos para generar el proceso (generate_process)
    process_max_length = 1500
    process_temperature = 0.55
    process_top_k = 40
    process_top_p = 0.8
    process_max_new_tokens=500

    # Hiperparámetros específicos para generar la propuesta de software (generate_proposal)
    proposal_max_length = 2000
    proposal_temperature = 0.8
    proposal_top_k = 60
    proposal_top_p = 0.9
    roposal_max_new_tokens=1000

    # Hiperparámetros específicos para generar historias de usuario (generate_user_stories)
    user_story_max_length = 2000
    user_story_temperature = 0.65
    user_story_top_k = 50
    user_story_top_p = 0.9
    user_stories_max_new_tokens=600

    chat_llm(
        generator, 
        prompt_template, 
        process_max_length, process_temperature, process_top_k, process_top_p,process_max_new_tokens,
        proposal_max_length, proposal_temperature, proposal_top_k, proposal_top_p,roposal_max_new_tokens,
        user_story_max_length, user_story_temperature, user_story_top_k, user_story_top_p,user_stories_max_new_tokens
    )

if __name__ == "__main__": 
    main(
        config_file="config.json", 
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", 
        pipeline_type="text-generation", 
        device="cpu"
    )
