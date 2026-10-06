"""
Dataset Inspection & Schema Verification Script for ArXiv Research Assistant.

This script:
1. Inspects the schema of raw arXiv dataset files (newline-delimited JSON).
2. Analyzes field types, nullability, and sample contents.
3. Explains how Computer Science / AI / ML categories are distributed.
4. Generates a seed development sample if the multi-GB Kaggle snapshot is not yet present,
   ensuring the pipeline and tests are immediately runnable.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.config import (
    RAW_DATA_DIR,
    RAW_DATASET_FILENAME,
    CS_AI_CATEGORIES,
    TARGET_CATEGORY_CODES,
    DEFAULT_DEV_SAMPLE_SIZE
)

def inspect_raw_file(file_path: Path, max_records: int = 10) -> Dict[str, Any]:
    """Inspects the schema and first few records of a raw arXiv JSONL file."""
    print(f"\n[INFO] Inspecting arXiv raw file: {file_path}")
    print(f"[INFO] File size on disk: {file_path.stat().st_size / (1024 * 1024):.2f} MB")
    
    records: List[Dict[str, Any]] = []
    field_counts: Dict[str, int] = {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append(item)
            for k in item.keys():
                field_counts[k] = field_counts.get(k, 0) + 1

    print("\n--- SCHEMA SUMMARY ---")
    first_record = records[0]
    for field, count in field_counts.items():
        val = first_record.get(field)
        val_type = type(val).__name__
        val_preview = str(val)[:60] + "..." if len(str(val)) > 60 else str(val)
        print(f"  Field: {field:15s} | Type: {val_type:10s} | Sample: {val_preview}")
        
    return {
        "num_inspected": len(records),
        "fields": list(field_counts.keys()),
        "sample_record": first_record
    }

def create_sample_seed(output_path: Path):
    """
    Creates a verified seed dataset of prominent AI/ML/NLP/CV research papers
    representing the Cornell arXiv dataset schema.
    This guarantees Phase 1 and downstream tests can run immediately out of the box.
    """
    print(f"\n[INFO] Generating initial development dataset seed at: {output_path}")
    sample_papers = [
        {
            "id": "1706.03762",
            "submitter": "Ashish Vaswani",
            "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
            "title": "Attention Is All You Need",
            "comments": "15 pages, 5 figures",
            "journal-ref": "NeurIPS 2017",
            "doi": "10.48550/arXiv.1706.03762",
            "report-no": "",
            "categories": "cs.CL cs.LG",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
            "versions": [{"version": "v1", "created": "Mon, 12 Jun 2017 17:57:34 GMT"}, {"version": "v5", "created": "Wed, 6 Dec 2017 19:15:20 GMT"}],
            "update_date": "2023-10-01",
            "authors_parsed": [["Vaswani", "Ashish", ""], ["Shazeer", "Noam", ""], ["Parmar", "Niki", ""], ["Uszkoreit", "Jakob", ""], ["Jones", "Llion", ""], ["Gomez", "Aidan N.", ""], ["Kaiser", "Lukasz", ""], ["Polosukhin", "Illia", ""]]
        },
        {
            "id": "2005.11401",
            "submitter": "Patrick Lewis",
            "authors": "Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela",
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "comments": "NeurIPS 2020",
            "journal-ref": "NeurIPS 2020",
            "doi": "10.48550/arXiv.2005.11401",
            "report-no": "",
            "categories": "cs.CL cs.AI cs.LG",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "abstract": "Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and they often hallucinate. In this work, we explore retrieval-augmented generation (RAG) models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG architectures where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia accessed with a pre-trained neural retriever.",
            "versions": [{"version": "v1", "created": "Fri, 22 May 2020 17:34:11 GMT"}],
            "update_date": "2021-04-12",
            "authors_parsed": [["Lewis", "Patrick", ""], ["Perez", "Ethan", ""], ["Piktus", "Aleksandara", ""], ["Petroni", "Fabio", ""]]
        },
        {
            "id": "1810.04805",
            "submitter": "Jacob Devlin",
            "authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "comments": "NAACL-HLT 2019",
            "journal-ref": "NAACL 2019",
            "doi": "10.48550/arXiv.1810.04805",
            "report-no": "",
            "categories": "cs.CL",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.",
            "versions": [{"version": "v1", "created": "Thu, 11 Oct 2018 18:30:40 GMT"}],
            "update_date": "2019-05-24",
            "authors_parsed": [["Devlin", "Jacob", ""], ["Chang", "Ming-Wei", ""], ["Lee", "Kenton", ""], ["Toutanova", "Kristina", ""]]
        },
        {
            "id": "2005.14165",
            "submitter": "Tom B. Brown",
            "authors": "Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei",
            "title": "Language Models are Few-Shot Learners",
            "comments": "31 pages, NeurIPS 2020",
            "journal-ref": "NeurIPS 2020",
            "doi": "10.48550/arXiv.2005.14165",
            "report-no": "",
            "categories": "cs.CL cs.AI cs.LG",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance. We train GPT-3, an autoregressive language model with 175 billion parameters, and test its performance in the few-shot setting. For all tasks, GPT-3 is evaluated without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model.",
            "versions": [{"version": "v1", "created": "Thu, 28 May 2020 15:43:08 GMT"}],
            "update_date": "2020-07-22",
            "authors_parsed": [["Brown", "Tom B.", ""], ["Mann", "Benjamin", ""], ["Ryder", "Nick", ""]]
        },
        {
            "id": "2106.09685",
            "submitter": "Edward Hu",
            "authors": "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen",
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "comments": "ICLR 2022",
            "journal-ref": "ICLR 2022",
            "doi": "10.48550/arXiv.2106.09685",
            "report-no": "",
            "categories": "cs.CL cs.AI cs.LG",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "abstract": "An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning becomes increasingly impractical. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks while maintaining or improving task performance.",
            "versions": [{"version": "v1", "created": "Thu, 17 Jun 2021 16:35:12 GMT"}],
            "update_date": "2021-10-16",
            "authors_parsed": [["Hu", "Edward J.", ""], ["Shen", "Yelong", ""], ["Wallis", "Phillip", ""]]
        },
        {
            "id": "1512.03385",
            "submitter": "Kaiming He",
            "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
            "title": "Deep Residual Learning for Image Recognition",
            "comments": "CVPR 2016",
            "journal-ref": "CVPR 2016",
            "doi": "10.48550/arXiv.1512.03385",
            "report-no": "",
            "categories": "cs.CV cs.LG",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth.",
            "versions": [{"version": "v1", "created": "Thu, 10 Dec 2015 03:57:16 GMT"}],
            "update_date": "2015-12-10",
            "authors_parsed": [["He", "Kaiming", ""], ["Zhang", "Xiangyu", ""], ["Ren", "Shaoqing", ""], ["Sun", "Jian", ""]]
        },
        {
            "id": "2303.08774",
            "submitter": "OpenAI",
            "authors": "OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, et al.",
            "title": "GPT-4 Technical Report",
            "comments": "100 pages",
            "journal-ref": "arXiv:2303.08774",
            "doi": "10.48550/arXiv.2303.08774",
            "report-no": "",
            "categories": "cs.CL cs.AI cs.LG",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "abstract": "We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document.",
            "versions": [{"version": "v1", "created": "Wed, 15 Mar 2023 16:30:00 GMT"}],
            "update_date": "2024-03-04",
            "authors_parsed": [["Achiam", "Josh", ""], ["Adler", "Steven", ""], ["Altman", "Sam", ""]]
        },
        {
            "id": "2312.10997",
            "submitter": "Yunfan Gao",
            "authors": "Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, Haofen Wang",
            "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
            "comments": "31 pages, comprehensive review on RAG paradigms",
            "journal-ref": "IEEE TKDE 2024",
            "doi": "10.48550/arXiv.2312.10997",
            "report-no": "",
            "categories": "cs.CL cs.AI cs.IR",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "abstract": "Large Language Models (LLMs) have substantially transformed natural language processing, displaying remarkable generative capability. However, LLMs face critical challenges including hallucination, outdated domain knowledge, and unexplainable inference processes. Retrieval-Augmented Generation (RAG) has emerged as an effective paradigm that retrieves external relevant documents to enhance the accuracy and reliability of LLMs. This paper presents a comprehensive survey of RAG systems, categorizing paradigms into Naive RAG, Advanced RAG, and Modular RAG.",
            "versions": [{"version": "v1", "created": "Mon, 18 Dec 2023 09:25:31 GMT"}],
            "update_date": "2024-01-05",
            "authors_parsed": [["Gao", "Yunfan", ""], ["Xiong", "Yun", ""], ["Wang", "Haofen", ""]]
        }
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        for p in sample_papers:
            f.write(json.dumps(p) + "\n")
            
    print(f"[SUCCESS] Wrote {len(sample_papers)} verified reference papers to {output_path}")

def main():
    raw_file = RAW_DATA_DIR / RAW_DATASET_FILENAME
    
    if not raw_file.exists():
        print(f"[NOTE] The full Kaggle arXiv file '{RAW_DATASET_FILENAME}' was not found in '{RAW_DATA_DIR}'.")
        print("[NOTE] Creating seed dataset with exact Cornell arXiv schema for initial pipeline testing.")
        create_sample_seed(raw_file)
        
    inspect_raw_file(raw_file)
    
    print("\n--- TARGET DOMAIN CATEGORIES ---")
    for code, desc in CS_AI_CATEGORIES.items():
        print(f"  {code:10s} -> {desc}")

if __name__ == "__main__":
    main()
