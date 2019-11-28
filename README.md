Save and Load transformer model:
                
        torch.save(model_to_save.state_dict(), os.path.join(OUTPUT_DIR,"pytorch_model.bin"))
        model_to_save.config.to_json_file(os.path.join(OUTPUT_DIR,"config.json"))
        tokenizer.save_vocabulary(OUTPUT_DIR)
        
        tokenizer = transformers.DistilBertTokenizer.from_pretrained(OUTPUT_DIR)
        model = transformers.DistilBertModel.from_pretrained(OUTPUT_DIR)

1. GPT-2:
  labels **are shifted** inside the model, i.e. you can set lm_labels = input_ids
                
        Example for GPT2DoubleHeadsModel:
        '''inputs (N,C,L)
           mc_token_ids (N,C) last token to use for cls
           mc_labels (N,)
           lm_labels (N,C,L)'''
        import torch
        from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
        
        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary
        
        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]
        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # where cls is to be used for classification
        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

2. Transformer XL:
               
        Transformer XL introduces recursive relationship between attention blocks to model dependency longer that one attention block. It also introduce relatives position embedding.
        mem_len should be set to smaller number during training
        same_length. Was set to True as default as original paper. Might be worthwhile to check otherwise on data. 
  
