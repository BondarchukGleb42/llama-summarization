import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id =  "SalmanFaroz/Llama-2-7b-QLoRa-samsum"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def predict(dscr, pos_comms, neg_comms):
    """
    Функция для суммаризации информации о товаре и поиску его плюсов и минусов.
    Принимает на вход:
        -dscr: str, текстовое описание товара;
        -pos_comms: list, список позитивных комментариев;
        -neg_comms: list, список негативных комментариев.
    Генерирует промпт для модели, получает ответ и парсит его. Возвращает dict вида:
    {
        'pluses': list, список позитивных характеристик товара;
        'minuses': list, список негативных характеристик товара.
    }
    """
    prompt = "\n".join(["Найди плюсы и минусы товара по его текстовому описанию и отзывам.",
                      "### Товар:",
                      dscr,
                      "\n"
                      "Позитивные отзывы: ",
                      "\n".join([f"{i+1}) {txt}" for i, txt in enumerate(pos_comms)]),
                      "\n"
                      "Негативные отзывы: ",
                      "".join([f"{i+1}) {txt}" for i, txt in enumerate(neg_comms)]),
                      "### Плюсы и минусы:"])
    
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=400,
        )[0],
        skip_special_tokens=True
    )
    output = output[len(prompt):].lower()
    if "плюсы" in output:
        pluses = output[output.index("плюсы")+7:]
        if "минусы" in output:
            pluses = pluses[:pluses.index("минусы")]
            pluses = pluses.split("\n")
            pluses = [x[3:] for i, x in enumerate(pluses) if len(x)>0 and str(i+1) in x]
        else:
            minuses = []
    else:
        pluses = []
            
    if "минусы" in output:
        minuses = output[output.index("минусы")+9:]
        minuses = minuses.split("\n")
        minuses = [x[3:] for i, x in enumerate(minuses) if len(x)>0 and str(i+1) in x]
    else:
        minuses = []
    
    return {"pluses": pluses, "minuses": minuses}
