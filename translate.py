#!/usr/bin/env python3
"""
Multi-Language Tafsir Translator
A Python script to translate Arabic or Urdu Quranic commentary (tafsir) to English
Automatically detects source language and translates using deep-translator
"""

import time
import re
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import json
from datetime import datetime

# Deep-translator imports
from deep_translator import GoogleTranslator

class TafsirTranslator:
    def __init__(self, delay_between_requests: float = 1.0):
        """
        Initialize the multi-language tafsir translator
        
        Args:
            delay_between_requests: Delay between API calls to respect rate limits
        """
        self.delay_between_requests = delay_between_requests
        self.supported_languages = {
            'ar': 'Arabic',
            'en': 'English',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'ur': 'Urdu'
        }
        # Dictionary to store extracted Quranic verses and their placeholders
        self.ayah_placeholders: Dict[str, str] = {}
        self.placeholder_counter: int = 0
        
        #logger.info("Multi-language Tafsir Translator initialized")
    
    def _reset_state(self):
        """Resets the ayah placeholder dictionary and counter for a new translation."""
        self.ayah_placeholders = {}
        self.placeholder_counter = 0

    def _extract_and_replace_ayahs(self, text: str) -> str:
        """
        Extracts Quranic verses from the text and replaces them with unique placeholders.
        Stores the original verses in self.ayah_placeholders.
        """
        # Regex to find text enclosed in "...", «...», or ﴾...﴿
        # This pattern prioritizes Quranic brackets but also covers standard quotes.
        # It's non-greedy to match the shortest possible string.
        ayah_pattern = re.compile(r'["«﴿](.*?)[»﴾"]', re.DOTALL)
        
        def replace_match(match):
            original_ayah = match.group(0)
            #unique_id = str(uuid.uuid4()).replace('-', '')
            placeholder = f"[{self.placeholder_counter}]"
            self.ayah_placeholders[placeholder] = original_ayah
            self.placeholder_counter += 1
            #logger.info(f"Extracted ayah: '{original_ayah}' -> Placeholder: '{placeholder}'")
            return placeholder
        
        processed_text = ayah_pattern.sub(replace_match, text)

        #logger.info(f"Ayahs extracted. Total placeholders created: {len(self.ayah_placeholders)}")
        return processed_text

    def _restore_ayahs(self, translated_text: str) -> str:
        """
        Restores the original Quranic verses into the translated text
        by replacing placeholders.
        """
        restored_text = translated_text

        sorted_placeholders = sorted(self.ayah_placeholders.items(), key=lambda item: len(item[0]), reverse=True)

        for placeholder_key, original_ayah_text in sorted_placeholders:
            restored_text = restored_text.replace(placeholder_key, original_ayah_text)

        #logger.info(f"Finished ayah restoration process.")
        return restored_text
    
    def preprocess_text(self, text: str, language: str) -> str:
        """
        Preprocess text
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        if language in ['ar', 'ur']:
            # Handle Arabic/Urdu punctuation
            text = re.sub(r'[«»]', '"', text)
            text = re.sub(r'[،]', ',', text)
            text = re.sub(r'[؛]', ';', text)
            text = re.sub(r'[؟]', '?', text)
            
            # Convert Arabic-Indic numerals to Western numerals
            arabic_nums = '٠١٢٣٤٥٦٧٨٩'
            english_nums = '0123456789'
            for ar_num, en_num in zip(arabic_nums, english_nums):
                text = text.replace(ar_num, en_num)
                
            # Handle Urdu-specific formatting
            if language == 'ur':
                # Remove excessive Urdu punctuation
                text = re.sub(r'[۔]+', '۔', text)  # Urdu full stop
                
        return text
    
    def split_text_intelligently(self, text: str, max_length: int = 3000) -> List[str]:
        """
        Split text into chunks prioritizing newlines, then by words, respecting max_length.
        
        Args:
            text: The input text to split.
            max_length: The maximum desired length for each chunk.
                        Note: Google Translate's recommended max is 5000 chars,
                        with advanced supporting up to 30000. 4000 is a safe middle ground.
        
        Returns:
            A list of text chunks.
        """
        if not text:
            return []
        
        chunks = []
        
        # 1. Split by newlines first to get logical blocks (paragraphs)
        # Using re.split to keep the newlines for potential reconstruction if needed later,
        # but for chunking, we process the actual text between newlines.
        paragraph_splits = re.split(r'(\n+)', text)
        current_chunk_buffer = [] # To build up chunks of words
        current_chunk_length = 0

        for i, segment in enumerate(paragraph_splits):
            if not segment.strip(): # Skip empty segments or pure whitespace/newlines
                # If it's a newline segment, consider it as a separator for chunks
                if segment.strip() == '' and '\n' in segment:
                    if current_chunk_buffer: # If there's content in the buffer, finalize it
                        chunks.append(" ".join(current_chunk_buffer).strip())
                        current_chunk_buffer = []
                        current_chunk_length = 0
                    # Add the newline itself as a "chunk" if we want to preserve exact spacing,
                    # but typically we rejoin with " " or "\n\n" later.
                    # For actual translation chunks, we don't send just newlines.
                continue

            # Process actual text content
            words = segment.split(' ') # Simple word split, can be refined for other languages

            for word in words:
                word_len = len(word)
                # Check if adding the next word (plus a space) exceeds max_length
                # If current_chunk_buffer is empty, just add the word
                # Otherwise, add word + space
                
                # Check if the current word itself is too long for a chunk (very rare, e.g., chemical names)
                if word_len > max_length:
                    #logger.warning(f"Very long word ({word_len} chars) encountered, splitting it arbitrarily.")
                    # Arbitrarily split the very long word
                    # This is a fallback for extreme cases, not ideal for linguistic sense
                    for j in range(0, word_len, max_length):
                        if word[j:j+max_length]:
                            chunks.append(word[j:j+max_length])
                    current_chunk_buffer = [] # Reset buffer after handling oversized word
                    current_chunk_length = 0
                    continue # Move to next word

                if current_chunk_buffer and (current_chunk_length + word_len + 1) > max_length: # +1 for space
                    # Current word won't fit, so finalize the current buffer
                    chunks.append(" ".join(current_chunk_buffer).strip())
                    current_chunk_buffer = [word]
                    current_chunk_length = word_len
                else:
                    # Add word to current buffer
                    current_chunk_buffer.append(word)
                    current_chunk_length += word_len + (1 if current_chunk_buffer else 0) # +1 for space if not first word

        # Add any remaining text in the buffer
        if current_chunk_buffer:
            chunks.append(" ".join(current_chunk_buffer).strip())
            
        # Filter out any empty chunks that might result from splitting logic
        chunks = [chunk for chunk in chunks if chunk]
        
        return chunks

    def translate_chunk(self, text: str, source_lang: str, target_lang: str, retry_count: int = 3) -> str:
        """
        Translate a single chunk with retry logic
        """
        for attempt in range(retry_count):
            try:
                # Add delay for rate limiting
                if attempt > 0:
                    time.sleep(self.delay_between_requests * (attempt + 1))
                
                # Create translator instance for this chunk
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                
                # Perform translation
                result = translator.translate(text)
                
                if result and result.strip():
                    return result.strip()
                #else:
                #    logger.warning(f"Empty result on attempt {attempt + 1}")
                    
            except Exception as e:
                #logger.warning(f"Translation attempt {attempt + 1} failed: {str(e)}")
                if attempt == retry_count - 1:
                    return f"[Translation failed: {str(e)}]"
                
                # Wait longer before retry
                time.sleep(self.delay_between_requests * 2)
                
        return "[Translation failed after all retries]"
    
    def translate_tafsir(self, 
                        input_text: str, 
                        source_language: str = "ar",
                        target_language: str = "en",
                        preserve_structure: bool = True) -> Dict[str, Union[str, int, List, float]]:
        """
        Main method to translate tafsir text,
		now preserving Quranic verses.
        
        Args:
            input_text: The text to translate
            source_language: manual language specification (ar/ur)
            preserve_structure: Whether to preserve text structure
        """
        #logger.info("Starting tafsir translation...")
        
        # Reset placeholders for a new translation session
        self._reset_state()

        # Step 1: Extract Ayats and insert placeholders
        text_with_placeholders = self._extract_and_replace_ayahs(input_text)
        #logger.info(f"Extracted {len(self.ayah_placeholders)} Quranic ayats and replaced with placeholders.")

        source_lang_name = self.supported_languages.get(source_language, source_language)
        target_lang_name = self.supported_languages.get(target_language, target_language)
        
        #logger.info(f"Source language: {source_lang_name} ({source_language})")
        #logger.info(f"Target language: {target_lang_name} ({target_language})")
        
        # Preprocess text
        processed_text = self.preprocess_text(text_with_placeholders, source_language)
        
        # Split into chunks
        chunks = self.split_text_intelligently(processed_text)
        #logger.info(f"Text split into {len(chunks)} chunks")
        
        translated_chunks = []
        failed_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            #logger.info(f"Translating chunk {i}/{len(chunks)}...")
            
            translated = self.translate_chunk(chunk, source_language, target_language)
            translated_chunks.append(translated)
            
            if translated.startswith("[Translation failed"):
                failed_chunks.append(i)
                #logger.error(f"Failed to translate chunk {i}")
            
            # Small delay between chunks
            time.sleep(self.delay_between_requests)
        
        # Combine translated chunks
        if preserve_structure:
            full_translation = " ".join(translated_chunks)
        else:
            full_translation = "\n\n".join(translated_chunks)
        
        # Step 2: Restore Ayats from placeholders
        final_translation = self._restore_ayahs(full_translation)
        # Post-process translation
        final_translation = self._post_process_translation(final_translation)
        
        # Calculate success metrics
        successful_chunks = len(chunks) - len(failed_chunks)
        success_rate = (successful_chunks / len(chunks)) * 100 if chunks else 0
        
        result = {
            'original_text': input_text,
            'processed_text_with_placeholders': processed_text,
            'translated_text': final_translation,
            'source_language_name': source_lang_name,
            'target_language_name': target_lang_name,
            'total_chunks': len(chunks),
            'successful_chunks': successful_chunks,
            'failed_chunks': failed_chunks,
            'success_rate': success_rate,
            'translation_timestamp': datetime.now().isoformat(),
			'ayah_preservation_details': self.ayah_placeholders,
            'chunks_detail': [
                {
                    'chunk_id': i,
                    'original': chunk,
                    'translated': trans,
                    'success': not trans.startswith("[Translation failed")
                }
                for i, (chunk, trans) in enumerate(zip(chunks, translated_chunks), 1)
            ]
        }
        
        #logger.info(f"Translation completed! Success rate: {success_rate:.1f}%")
        return result
    
    def _post_process_translation(self, text: str) -> str:
        """Post-process translation for better readability"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation spacing - remove spaces before punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)

        # Handle hyphens specifically - remove extra spaces around hyphens but keep single spaces
        text = re.sub(r'\s+-\s+', '-', text)  # Remove spaces around hyphens
        text = re.sub(r'(\w)\s+-', r'\1-', text)  # Remove space before hyphen after word
        text = re.sub(r'-\s+(\w)', r'-\1', text)  # Remove space after hyphen before word
            
        # Ensure proper spacing after sentence-ending punctuation
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Capitalize sentences
        sentences = re.split(r'([.!?]+)', text)
        processed_sentences = []

        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip() if i < len(sentences) else ''
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''

            if sentence:
                # Check if the sentence starts with an ayah to avoid capitalizing it
                if not any(sentence.startswith(ayah[:10]) for ayah in self.ayah_placeholders.values()):
                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()

                processed_sentences.append(sentence)
                if punctuation:
                    processed_sentences.append(punctuation)
                    # Add space after punctuation if there's more content
                    if i + 2 < len(sentences) and sentences[i + 2].strip():
                        processed_sentences.append(' ')
        
        return ''.join(processed_sentences).strip()

def translate_text(text: str, source_lang: str = "en", target_lang: str = "ar"):
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    return translator.translate(text=text)