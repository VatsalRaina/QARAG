# Question-Based Retrieval using Atomic Units for Enterprise RAG

This repository contains the code accompanying the paper **Question-Based Retrieval using Atomic Units for Enterprise RAG**, accepted to the **FEVER workshop at EMNLP 2024**.

## Paper Overview

Enterprise retrieval-augmented generation (RAG) provides a flexible framework for combining large language models (LLMs) with internal, potentially time-sensitive documents. In a typical RAG setup, documents are chunked, relevant chunks are retrieved for a user query, and these chunks are passed to an LLM to generate a response. 

However, the retrieval process can limit performance, as incorrect chunks can cause the LLM to generate inaccurate responses. This work introduces a zero-shot approach to improve retrieval accuracy by decomposing chunks into atomic statements and generating synthetic questions based on those atomic units. Retrieval is then performed by finding the closest synthetic questions and associated chunks to the user query. This method shows higher recall than traditional chunk-based retrieval, leading to better performance in the RAG pipeline.

## Note

Please note that this codebase is under construction to make it more intuitive and user-friendly. We are working on improving the structure and usability. However, the core components necessary to reproduce the experiments from the paper are fully functional and can be found in the qarag/zero-shot/ directory.

