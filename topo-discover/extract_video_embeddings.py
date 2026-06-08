import torch
import cv2
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoProcessor, AutoModel
from qwen_omni_utils import process_mm_info
import torch.nn.functional as F
import json

class OmniEmbedExtractor:
    def __init__(self, model_name="nvidia/omni-embed-nemotron-3b", device="cuda:0"):
        """
        Initialize NVIDIA Omni-Embed-Nemotron model.
        This model is specifically designed for extracting embeddings.
        """
        print(f"Loading embedding model: {model_name}")
        self.device = device
        
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    def split_video_by_time(self, video_path, chunk_duration=2.0, output_dir=None):
        """
        Split video file into temporal chunks using ffmpeg.
        
        Args:
            video_path: Path to video file
            chunk_duration: Duration of each chunk in seconds
            output_dir: Directory to save chunks (temp dir if None)
        
        Returns:
            List of paths to chunk files
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = Path(video_path)
        
        # Get video duration
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
        
        chunk_files = []
        chunk_idx = 0
        start_time = 0.0
        accum_time = chunk_duration
        
        while accum_time < duration:

            # end_time = min(start_time + chunk_duration, duration)
            
            end_time = min(start_time + accum_time, duration)
            chunk_file = output_dir / f"{video_path.stem}_chunk_{chunk_idx:04d}.mp4"
            
            # Extract chunk using ffmpeg
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-ss', str(start_time),
                # '-t', str(chunk_duration),
                '-t', str(accum_time),  # Use accum_time to ensure we cover the whole video without gaps
                '-c', 'copy',  # Copy streams without re-encoding (faster)
                str(chunk_file)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            chunk_files.append({
                'path': chunk_file,
                'chunk_id': chunk_idx,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            })
            
            chunk_idx += 1
            # start_time = end_time
            accum_time += chunk_duration
        
        return chunk_files
    
    def extract_embedding(self, video_path, use_audio=False, prompt=None):
        """
        Extract embedding from a video chunk.
        
        Args:
            video_path: Path to video file
            use_audio: Whether to use audio from video
            prompt: Optional text prompt to guide embedding
        
        Returns:
            numpy array of shape (2048,) - the embedding vector
        """
        # Prepare input format
        content = []
        
        if prompt:
            content.append({"type": "text", "text": f"passage: {prompt}"})
        
        content.append({"type": "video", "video": str(video_path)})
        
        if use_audio:
            content.append({"type": "audio", "audio": str(video_path)})
        
        documents = [{"role": "user", "content": content}]
        
        # Process inputs
        documents_texts = self.processor.apply_chat_template(
            documents, 
            add_generation_prompt=False, 
            tokenize=False
        )
        
        audio, images, videos = process_mm_info(
            documents, 
            use_audio_in_video=use_audio
        )
        
        videos_kwargs = {
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 2.0  # Process at 2 FPS
        }
        
        inputs = self.processor(
            text=documents_texts,
            audio=audio,
            images=images,
            videos=videos,
            videos_kwargs=videos_kwargs,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get the embedding from hidden states (tuple of all layers)
            # hidden_states[-1] is the last layer's hidden state
            hidden_states = outputs.hidden_states[-1]  # Shape: (batch, seq_len, hidden_dim)
            embeddings = hidden_states[:, -1, :]  # Take last token
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings.cpu().float().numpy()[0]  # Shape: (2048,)
    
    def process_video_chunks(self, video_path, chunk_duration=2.0, 
                           use_audio=True, output_dir=None, 
                           keep_chunks=False, prompt=None):
        """
        Process video by splitting into chunks and extracting embeddings.
        
        Args:
            video_path: Path to video file
            chunk_duration: Duration of each chunk in seconds
            use_audio: Whether to include audio
            output_dir: Directory to save embeddings
            keep_chunks: Whether to keep video chunk files
            prompt: Optional text prompt for contextualized embeddings
        
        Returns:
            Dictionary with embeddings and metadata
        """
        video_path = Path(video_path)
        
        # Create temporary directory for chunks
        temp_dir = tempfile.mkdtemp() if not keep_chunks else output_dir
        
        print(f"Splitting video into {chunk_duration}s chunks...")
        chunks = self.split_video_by_time(video_path, chunk_duration, temp_dir)
        
        print(f"Extracting embeddings from {len(chunks)} chunks...")
        
        results = []
        for chunk_info in chunks:
            print(f"Processing chunk {chunk_info['chunk_id']}...", end='\r')
            
            embedding = self.extract_embedding(
                chunk_info['path'],
                use_audio=use_audio,
                prompt=prompt
            )
            
            results.append({
                'chunk_id': chunk_info['chunk_id'],
                'start_time': chunk_info['start_time'],
                'end_time': chunk_info['end_time'],
                'duration': chunk_info['duration'],
                'embedding': embedding,
                'embedding_dim': embedding.shape[0]
            })
            
            # Clean up chunk file if not keeping
            if not keep_chunks:
                chunk_info['path'].unlink()
        
        print(f"\nCompleted {len(results)} chunks")
        
        output_data = {
            'video_path': str(video_path),
            'num_chunks': len(results),
            'chunk_duration': chunk_duration,
            'use_audio': use_audio,
            'model': 'nvidia/omni-embed-nemotron-3b',
            'embedding_dim': 2048,
            'chunks': results
        }
        
        # Save embeddings
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as compressed numpy arrays
            embeddings_array = np.array([chunk['embedding'] for chunk in results])
            output_file = output_dir / f"{video_path.stem}_nemotron_embeddings.npz"
            
            # Create chunk file paths for reference
            chunk_paths = []
            if keep_chunks:
                chunk_paths = [str(chunks[i]['path']) for i in range(len(results))]
            
            np.savez_compressed(
                output_file,
                embeddings=embeddings_array,
                timestamps=np.array([[c['start_time'], c['end_time']] for c in results]),
                chunk_ids=np.array([c['chunk_id'] for c in results]),
                chunk_paths=np.array(chunk_paths) if chunk_paths else None,
                video_name=video_path.stem
            )
            
            # Save metadata
            metadata_file = output_dir / f"{video_path.stem}_metadata.json"
            json_data = output_data.copy()
            for i, chunk in enumerate(json_data['chunks']):
                chunk['embedding'] = f"Saved in {output_file.name}"
                if keep_chunks and i < len(chunk_paths):
                    chunk['chunk_path'] = chunk_paths[i]
            
            with open(metadata_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"Saved embeddings to {output_file}")
            print(f"Shape: {embeddings_array.shape}")
        
        return output_data


# Example usage
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Extract video embeddings')
    parser.add_argument('--group-index', type=int, default=0, help='Index of video group to process')
    args = parser.parse_args()

    # Initialize extractor
    extractor = OmniEmbedExtractor(model_name="nvidia/omni-embed-nemotron-3b")
    
    # # Process single video
    # video_path = "task-alignvideo/ses-01_run-01_order-01_content-idiots.mp4"
    
    # results = extractor.process_video_chunks(
    #     video_path=video_path,
    #     chunk_duration=2.0,
    #     use_audio=True,
    #     output_dir="embeddings_output_v2",
    #     keep_chunks=True,  # Keep video chunk files for later analysis
    # )
    
    # print(f"\nExtracted {results['num_chunks']} embeddings")
    # print(f"Each embedding has {results['embedding_dim']} dimensions")
    
    # Batch process multiple videos from Spacetop dataset
    print("\n" + "="*50)
    print("Batch Processing")
    print("="*50)
    
    video_dir = Path("task-alignvideo")
    output_base = Path("spacetop_embeddings_v2")

    video_files = video_dir.glob("*.mp4")

    # chunk video files into 10 groups for parallel processing
    num_groups = 10
    video_files = sorted(video_files)
    groups = [video_files[i::num_groups] for i in range(num_groups)]
    video_files = groups[args.group_index]
    
    for video_file in sorted(video_files):
        print(f"\nProcessing: {video_file.name}")
        try:
            results = extractor.process_video_chunks(
                video_path=video_file,
                chunk_duration=2.0,
                use_audio=True,
                output_dir=output_base,
                keep_chunks=True  # Keep chunks for later analysis
            )
            print(f"✓ Completed {video_file.name}")
        except Exception as e:
            print(f"✗ Error processing {video_file.name}: {e}")
            continue
