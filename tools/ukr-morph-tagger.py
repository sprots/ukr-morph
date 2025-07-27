import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from pathlib import Path
import re
from TorchCRF import CRF
import warnings
warnings.filterwarnings('ignore')

# Ukrainian vowels for additional features
UKRAINIAN_VOWELS = {'а', 'е', 'є', 'и', 'і', 'ї', 'о', 'у', 'ю', 'я'}

# Ukrainian POS tags from training script
UKRAINIAN_POS_TAGS = [
    'N', 'V', 'A', 'R', 'P', 'D', 'M', 'S', 'Cc', 'Cs', 'Q',
    'I', 'Z', 'Y', 'X', 'Va', 'Np'
]

# Enhanced CNN model (same as in training script)
class EnhancedCharCNNTagger(nn.Module):
    def __init__(self, vocab_size, label_size, pos_size=None, embedding_dim=128, 
                 cnn_filters=[256, 192, 128], kernel_sizes=[5, 5, 5], 
                 lstm_hidden=256, use_lstm=True, use_crf=True, dropout=0.3):
        super(EnhancedCharCNNTagger, self).__init__()
        self.use_lstm = use_lstm
        self.use_crf = use_crf
        self.use_pos = pos_size is not None
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # POS embedding if available
        if self.use_pos:
            self.pos_embedding = nn.Embedding(pos_size, 32)
            total_embedding_dim = embedding_dim + 32 + 1  # +1 for vowel feature
        else:
            total_embedding_dim = embedding_dim + 1  # +1 for vowel feature
        
        # Multi-layer CNN with different kernel sizes
        self.cnn_layers = nn.ModuleList()
        input_dim = total_embedding_dim
        
        for i, (filters, kernel_size) in enumerate(zip(cnn_filters, kernel_sizes)):
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(input_dim, filters, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
            ))
            input_dim = filters
        
        # LSTM layer
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=lstm_hidden,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_hidden > 1 else 0
            )
            linear_input_size = 2 * lstm_hidden
        else:
            linear_input_size = input_dim
        
        # Output layer
        self.fc = nn.Linear(linear_input_size, label_size)
        self.dropout = nn.Dropout(dropout)
        
        # CRF layer
        if use_crf:
            self.crf = CRF(label_size, batch_first=True)

    def _extract_features(self, x, pos_tags=None):
        """Extract linguistic features for characters"""
        batch_size, seq_len = x.shape
        
        # Character embeddings
        char_embed = self.char_embedding(x)
        
        # Vowel features
        vowel_features = torch.zeros(batch_size, seq_len, 1, device=x.device)
        for i in range(batch_size):
            for j in range(seq_len):
                char_idx = x[i, j].item()
                if char_idx > 0:  # Not padding
                    # Get character from reverse mapping
                    char = next((c for c, idx in self.char2idx.items() if idx == char_idx), None)
                    if char and char in UKRAINIAN_VOWELS:
                        vowel_features[i, j, 0] = 1.0
        
        features = [char_embed, vowel_features]
        
        # POS features if available
        if self.use_pos and pos_tags is not None:
            pos_embed = self.pos_embedding(pos_tags)
            # Expand POS embedding to match sequence length
            pos_embed = pos_embed.unsqueeze(1).expand(-1, seq_len, -1)
            features.append(pos_embed)
        
        return torch.cat(features, dim=-1)

    def forward(self, x, pos_tags=None, mask=None):
        # Extract features
        features = self._extract_features(x, pos_tags)
        
        # CNN processing
        cnn_input = features.permute(0, 2, 1)  # (batch, features, seq)
        
        for cnn_layer in self.cnn_layers:
            cnn_input = cnn_layer(cnn_input)
        
        cnn_output = cnn_input.permute(0, 2, 1)  # (batch, seq, features)
        
        # LSTM processing
        if self.use_lstm:
            lstm_output, _ = self.lstm(cnn_output)
            features = self.dropout(lstm_output)
        else:
            features = cnn_output
        
        # Final classification
        emissions = self.fc(features)
        
        # CRF decoding during inference
        if self.use_crf and not self.training and mask is not None:
            return self.crf.decode(emissions, mask)
        
        return emissions

    def compute_loss(self, emissions, tags, mask=None):
        """Compute loss (CRF or CrossEntropy)"""
        if self.use_crf:
            return -self.crf(emissions, tags, mask=mask, reduction='mean')
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            return criterion(emissions.view(-1, emissions.shape[-1]), tags.view(-1))

class LemmaDataset(Dataset):
    def __init__(self, lemmas, pos_tags=None):
        self.lemmas = lemmas
        self.pos_tags = pos_tags
        
    def __len__(self):
        return len(self.lemmas)
    
    def __getitem__(self, idx):
        if self.pos_tags:
            return self.lemmas[idx], self.pos_tags[idx]
        return self.lemmas[idx]

def extract_pos_tag(row):
    """Extract POS tag using the specified algorithm"""
    # First try multext field
    multext = str(row['multext'])
    if multext and multext != 'nan' and multext != '':
        # First letter or 2 letters for Np, Cc, Cs
        if multext.startswith(('Np', 'Cc', 'Cs')):
            pos = multext[:2]
        else:
            pos = multext[0]
        
        if pos in UKRAINIAN_POS_TAGS:
            return pos
        else:
            print(f"Unknown POS tag from multext: {pos} in {multext}")
    
    # If UNK, try freq field (last letters after 2 digits)
    freq = str(row['freq'])
    if freq and freq != 'nan' and freq != '0':
        match = re.search(r'\d{2}([A-Za-z]+)$', freq)
        if match:
            pos = match.group(1)
            if pos in UKRAINIAN_POS_TAGS:
                return pos
            else:
                print(f"Unknown POS tag from freq: {pos} in {freq}")
    
    # Finally try pos field
    pos_field = str(row['pos'])
    if pos_field and pos_field != 'nan':
        # If there are 2 pos tags separated by ; choose first
        pos = pos_field.split(';')[0].strip()
        if pos in UKRAINIAN_POS_TAGS:
            return pos
        else:
            print(f"Unknown POS tag from pos field: {pos} in {pos_field}")
    
    # Default to 'X' if nothing found
    return 'X'

def encode_lemma(lemma, char2idx):
    """Encode lemma to character indices"""
    unknown_chars = []
    encoded = []
    
    for char in lemma:
        if char in char2idx:
            encoded.append(char2idx[char])
        else:
            encoded.append(char2idx.get('<UNK>', 0))
            unknown_chars.append(char)
    
    return encoded, unknown_chars

def pad_batch(batch_data, pad_idx=0):
    """Pad sequences in batch"""
    if not batch_data:
        return torch.tensor([], dtype=torch.long)
    
    max_len = max(len(seq) for seq in batch_data)
    padded = torch.full((len(batch_data), max_len), pad_idx, dtype=torch.long)
    
    for i, seq in enumerate(batch_data):
        padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    
    return padded

def create_attention_mask(sequences, pad_idx=0):
    """Create attention mask for sequences"""
    return (sequences != pad_idx)

def bio_to_morphemes(lemma, bio_tags, idx2label):
    """Convert BIO tags to morpheme:tag format"""
    if not lemma or not bio_tags:
        return f"{lemma}:R"  # Fallback to root
    
    morphemes = []
    current_morpheme = ""
    current_tag = None
    
    # Tag mapping for output
    tag_map = {
        'B-P': 'P', 'I-P': 'P',  # Prefix
        'B-R': 'R', 'I-R': 'R',  # Root
        'B-S': 'S', 'I-S': 'S',  # Suffix
        'B-F': 'F', 'I-F': 'F',  # Flexion
        'B-X': 'X', 'I-X': 'X',  # Other
        'B-I': 'I', 'I-I': 'I',  # Infix
        'B-H': 'H', 'I-H': 'H',  # Hypermorph
        'O': 'O'  # Outside
    }
    
    for i, (char, tag_idx) in enumerate(zip(lemma, bio_tags)):
        if tag_idx >= len(idx2label):
            tag = 'O'
        else:
            tag = idx2label[tag_idx]
        
        if tag.startswith('B-') or tag == 'O':
            # Save previous morpheme if exists
            if current_morpheme:
                morpheme_tag = tag_map.get(current_tag, 'R')
                morphemes.append(f"{current_morpheme}:{morpheme_tag}")
            
            # Start new morpheme
            current_morpheme = char
            current_tag = tag
        elif tag.startswith('I-') and current_tag and tag[2:] == current_tag[2:]:
            # Continue current morpheme
            current_morpheme += char
        else:
            # Handle inconsistent tagging
            if current_morpheme:
                morpheme_tag = tag_map.get(current_tag, 'R')
                morphemes.append(f"{current_morpheme}:{morpheme_tag}")
            current_morpheme = char
            current_tag = tag
    
    # Add final morpheme
    if current_morpheme:
        morpheme_tag = tag_map.get(current_tag, 'R')
        morphemes.append(f"{current_morpheme}:{morpheme_tag}")
    
    return "/".join(morphemes) if morphemes else f"{lemma}:R"

def calculate_confidence(emissions, predictions, mask=None):
    """Calculate confidence score from model outputs"""
    if mask is not None:
        # For CRF models, we don't have direct probabilities
        # Use a simple heuristic based on emission scores
        confidence_scores = []
        for i, (emission, pred_seq) in enumerate(zip(emissions, predictions)):
            valid_length = mask[i].sum().item()
            seq_confidence = []
            for j in range(valid_length):
                if j < len(pred_seq):
                    # Get the emission score for the predicted tag
                    pred_tag = pred_seq[j]
                    if pred_tag < emission.shape[1]:
                        score = torch.softmax(emission[j], dim=0)[pred_tag].item()
                        seq_confidence.append(score)
            
            if seq_confidence:
                confidence_scores.append(np.mean(seq_confidence))
            else:
                confidence_scores.append(0.0)
        
        return confidence_scores
    else:
        # For non-CRF models, use softmax probabilities
        probs = torch.softmax(emissions, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        
        confidence_scores = []
        for i, prob_seq in enumerate(max_probs):
            # Calculate mean confidence for non-padding positions
            non_pad_mask = (predictions[i] != 0)
            if non_pad_mask.any():
                confidence = prob_seq[non_pad_mask].mean().item()
                confidence_scores.append(confidence)
            else:
                confidence_scores.append(0.0)
        
        return confidence_scores

def collate_fn(batch, char2idx, pos2idx=None):
    """Collate function for DataLoader"""
    if pos2idx:
        lemmas, pos_tags = zip(*batch)
        pos_indices = [pos2idx.get(pos, 0) for pos in pos_tags]
        pos_tensor = torch.tensor(pos_indices, dtype=torch.long)
    else:
        lemmas = batch
        pos_tensor = None
    
    # Encode lemmas
    encoded_lemmas = []
    all_unknown_chars = []
    
    for lemma in lemmas:
        encoded, unknown_chars = encode_lemma(lemma, char2idx)
        encoded_lemmas.append(encoded)
        all_unknown_chars.append(unknown_chars)
    
    return encoded_lemmas, pos_tensor, all_unknown_chars

def load_model(model_path):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model_config = checkpoint['model_config']
    char2idx = checkpoint['char2idx']
    label2idx = checkpoint['label2idx']
    idx2label = checkpoint['idx2label']
    pos2idx = checkpoint.get('pos2idx', None)
    
    # Create model
    model = EnhancedCharCNNTagger(
        vocab_size=model_config['vocab_size'],
        label_size=model_config['label_size'],
        pos_size=model_config.get('pos_size', None),
        use_lstm=model_config['use_lstm'],
        use_crf=model_config['use_crf']
    )
    
    # Add char2idx to model for feature extraction
    model.char2idx = char2idx
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, char2idx, label2idx, idx2label, pos2idx

def process_csv(input_file, output_file, model_path, batch_size=32):
    """Process CSV file with morpheme tagging"""
    print(f"Loading model from {model_path}")
    model, char2idx, label2idx, idx2label, pos2idx = load_model(model_path)
    
    print(f"Loading CSV file: {input_file}")
    df = pd.read_csv(input_file)
    
    if 'lemma' not in df.columns:
        raise ValueError("CSV file must contain 'lemma' column")
    
    # Extract POS tags
    print("Extracting POS tags...")
    pos_tags = df.apply(extract_pos_tag, axis=1).tolist()
    
    # Prepare data
    lemmas = df['lemma'].astype(str).tolist()
    
    # Create dataset
    if pos2idx:
        dataset = LemmaDataset(lemmas, pos_tags)
        def batch_collate(batch):
            return collate_fn(batch, char2idx, pos2idx)
    else:
        dataset = LemmaDataset(lemmas)
        def batch_collate(batch):
            return collate_fn(batch, char2idx)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_collate)
    
    # Process batches
    print("Processing lemmas...")
    all_predictions = []
    all_confidences = []
    all_errors = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if len(batch_data) == 3:
                encoded_lemmas, pos_tensor, unknown_chars_batch = batch_data
            else:
                encoded_lemmas, unknown_chars_batch = batch_data
                pos_tensor = None
            
            if not encoded_lemmas:
                continue
            
            # Pad sequences
            padded_input = pad_batch(encoded_lemmas, pad_idx=0)
            
            if padded_input.size(0) == 0:
                continue
            
            # Create mask
            mask = create_attention_mask(padded_input) if model.use_crf else None
            
            try:
                # Forward pass
                if model.use_crf:
                    predictions = model(padded_input, pos_tensor, mask=mask)
                    # Get emissions for confidence calculation
                    emissions = model(padded_input, pos_tensor)
                    confidences = calculate_confidence(emissions, predictions, mask)
                else:
                    logits = model(padded_input, pos_tensor)
                    predictions = logits.argmax(-1)
                    confidences = calculate_confidence(logits, predictions)
                
                # Convert predictions to morpheme format
                batch_start = batch_idx * batch_size
                for i, (pred_seq, unknown_chars) in enumerate(zip(predictions, unknown_chars_batch)):
                    lemma_idx = batch_start + i
                    if lemma_idx >= len(lemmas):
                        break
                    
                    lemma = lemmas[lemma_idx]
                    
                    # Handle unknown characters
                    if unknown_chars:
                        error_msg = f"Unknown characters in '{lemma}': {', '.join(set(unknown_chars))}"
                        all_errors.append(error_msg)
                        print(f"Warning: {error_msg}")
                    
                    # Convert to morpheme format
                    if isinstance(pred_seq, list):
                        # CRF output
                        valid_length = len(lemma)
                        pred_tags = pred_seq[:valid_length]
                    else:
                        # Regular output
                        valid_length = len(lemma)
                        pred_tags = pred_seq[:valid_length].tolist()
                    
                    morpheme_tags = bio_to_morphemes(lemma, pred_tags, idx2label)
                    all_predictions.append(morpheme_tags)
                    
                    # Add confidence
                    if i < len(confidences):
                        all_confidences.append(confidences[i])
                    else:
                        all_confidences.append(0.0)
                        
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                # Fallback: tag all as root
                batch_start = batch_idx * batch_size
                for i in range(len(encoded_lemmas)):
                    lemma_idx = batch_start + i
                    if lemma_idx >= len(lemmas):
                        break
                    lemma = lemmas[lemma_idx]
                    all_predictions.append(f"{lemma}:R")
                    all_confidences.append(0.0)
    
    # Ensure we have predictions for all lemmas
    while len(all_predictions) < len(lemmas):
        lemma = lemmas[len(all_predictions)]
        all_predictions.append(f"{lemma}:R")
        all_confidences.append(0.0)
    
    # Add predictions to dataframe
    df['predicted'] = all_predictions[:len(df)]
    df['confidence'] = all_confidences[:len(df)]
    
    # Save results
    print(f"Saving results to {output_file}")
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\nProcessed {len(df)} lemmas")
    print(f"Errors encountered: {len(all_errors)}")
    if all_errors:
        print("First 5 errors:")
        for error in all_errors[:5]:
            print(f"  - {error}")
    
    print(f"Average confidence: {np.mean(all_confidences):.4f}")
    print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Tag morphemes in CSV lemmas using trained model")
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_csv", help="Output CSV file path")
    parser.add_argument("model_path", help="Path to trained model file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file '{args.input_csv}' not found")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found")
        return
    
    # Process CSV
    process_csv(args.input_csv, args.output_csv, args.model_path, args.batch_size)

if __name__ == "__main__":
    main()