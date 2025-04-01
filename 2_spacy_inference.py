import spacy
import os
import traceback
from transformers import AutoTokenizer # Import AutoTokenizer

# --- Configuration details ---
hf_model_name = "HUMADEX/greek_medical_ner"
language_code = "el"

# --- !!! ADD THIS CHECK !!! ---
print(f"\n--- Checking tokenizer loading for '{hf_model_name}' directly ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    if tokenizer:
        print("Successfully loaded tokenizer directly using transformers.")
        # print(tokenizer) # Optional: print tokenizer info
    else:
        # Should technically raise an error if not found, but good to check
        print("WARNING: AutoTokenizer.from_pretrained returned None or empty.")
except Exception as e:
    print(f"ERROR: Failed to load tokenizer directly using transformers: {e}")
    print("This suggests an issue with accessing the model from Hugging Face Hub or a problem with the 'transformers' library installation.")
    print("Stopping execution.")
    exit() # Exit if the basic tokenizer loading fails
print("--- Tokenizer check complete ---\n")
# --- END OF CHECK ---


print(f"Attempting to build spaCy pipeline programmatically for model: {hf_model_name}")

try:
    # 1. Create a blank pipeline for the target language
    nlp = spacy.blank(language_code)
    print(f"Created blank '{language_code}' pipeline.")

    # 2. Add the transformer component
    transformer_config = {
        "model": {"name": hf_model_name}
        # You might try adding explicit tokenizer config here if the direct check passes but spaCy fails
        # "tokenizer_config": {"use_fast": True} # Example option
    }
    nlp.add_pipe("transformer", config=transformer_config)
    print(f"Added 'transformer' component configured with model '{hf_model_name}'.")

    # 3. Add the NER component
    nlp.add_pipe("ner")
    print("Added 'ner' component. It should use the transformer's NER head.")

    print("\nPipeline built successfully:")
    print(nlp.pipe_names)

    # --- Initialize the pipeline (IMPORTANT for transformers) ---
    # Processing text will initialize components, but explicit initialization is clearer
    # and can catch errors earlier.
    print("\nInitializing the pipeline...")
    nlp.initialize() # Add this line
    print("Pipeline initialized.")
    # --- End Initialization ---

    # --- Test with some sample Greek legal text ---
    text_el = """
    Άρθρο 1: Ορισμοί
    Για τους σκοπούς της παρούσας σύμβασης, οι ακόλουθοι όροι έχουν την έννοια που τους αποδίδεται κατωτέρω:
    "Εταιρεία": Η Αλφα Α.Ε., με έδρα την Αθήνα, οδός Ερμού 10.
    "Πελάτης": Η Βήτα Ε.Π.Ε., με έδρα τη Θεσσαλονίκη.
    "Έργο": Η ανάπτυξη λογισμικού σύμφωνα με το Παράρτημα Α.
    Ημερομηνία Έναρξης: 1η Ιανουαρίου 2024.
    """

    print("\nProcessing text...")
    doc = nlp(text_el) # Now process the text
    print("Text processing complete.")

    print("\n--- Detected Entities ---")
    if not doc.ents:
        print("No entities detected.")
    for ent in doc.ents:
        print(f"Text: '{ent.text}', Label: {ent.label_}, Start: {ent.start_char}, End: {ent.end_char}")

    if "ner" in nlp.pipe_names:
      ner_labels = nlp.get_pipe("ner").labels
      print("\n--- NER Labels defined in the loaded model ---")
      print(ner_labels)
    else:
      print("\nNER component not found in pipeline.")


except ImportError as e:
    # ... (keep existing error handling) ...
    print(f"\nIMPORT ERROR: {e}")
    print("Make sure you have installed spacy, torch/tensorflow, transformers, and spacy-transformers:")
    print("pip install -U spacy spacy-transformers torch transformers") # Or tensorflow
except Exception as e:
    # ... (keep existing error handling) ...
    print(f"\n--- An unexpected error occurred ---")
    print(f"Error message: {e}")
    print("\n--- Traceback ---")
    traceback.print_exc()
    print("--- End Traceback ---\n")