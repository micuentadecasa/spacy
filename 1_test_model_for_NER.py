from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import traceback

# model_name = "amichailidis/greek_legal_bert_v2-finetuned-ner" - only a entity
model_name = "joelniklaus/legal-greek-roberta-base"
# model_name = "alexaapo/greek_legal_bert_v2" # prepared for fine tuning
# model_name = "amichailidis/bert-base-greek-uncased-v1-finetuned-ner"- only a entity
print(f"\n--- Verifying model '{model_name}' directly with transformers ---")

try:
    # Load tokenizer and model specifically for Token Classification (NER)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # --- CRITICAL CHECK: Inspect the model's configuration ---
    print("Model Config (first few keys):", list(model.config.to_dict().keys())[:10])
    if hasattr(model.config, 'id2label'):
        print("Labels found in HF config (id2label):", model.config.id2label)
        if not model.config.id2label:
           print("WARNING: id2label mapping is empty!")
    else:
        print("ERROR: No 'id2label' mapping found in model config. This is likely not a usable NER model.")
        exit() # Exit if no labels fundamentally exist

    # Create a basic NER pipeline with transformers
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

    text_el = """
    Η Αλφα Α.Ε., με έδρα την Αθήνα, και η Βήτα Ε.Π.Ε. συμφωνούν.
    """
    results = ner_pipeline(text_el)

    print("\n--- Results from transformers NER pipeline ---")
    if not results:
        print("No entities detected by transformers pipeline.")
    for entity in results:
        print(entity)

except Exception as e:
    print(f"\n--- Error during transformers direct verification ---")
    print(f"Error message: {e}")
    print("\n--- Traceback ---")
    traceback.print_exc()
    print("--- End Traceback ---\n")

print("--- Direct verification complete ---")