# Corpus

The files here are **written for development**, not real policy documents. They
imitate the register and structure of Indian health-insurance wordings — waiting
periods, sub-limits, exclusions, claim procedure — so the pipeline can be
exercised end to end without a download.

## Using real documents

Drop `.pdf` files into this directory and rebuild. Ingestion handles PDFs via
PyMuPDF:

    pip install -e "examples/reference_system[pdf]"

Public sources: IRDAI-registered insurers publish policy wordings and customer
information sheets on their own sites. Check each insurer's terms before
redistributing anything in a public repo — ingest locally and keep the corpus
directory gitignored if in doubt.

## What matters for evaluation

Chunk ids are content-addressed (`doc:sha1`), so re-ingesting unchanged text
yields the same ids and existing gold references stay valid. Editing a document
changes only the ids of the chunks whose text changed.
