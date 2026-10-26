python .\check_accuracy.py# CampusMail — College Email Categorization

## Overview
CampusMail is a Python project that categorizes college emails using
rule-based classification. It produces a categorized CSV and a separate
queue of emails that may need manual review.

## Main Features
- Rule-based email categorization
- Multiple college email categories
- Manual-review queue for ambiguous messages
- CSV-based input and output
- Validation using manually verified email labels

## Project Structure
- `src/auto_categorize.py` — main categorization script
- `src/auto_categorize_backup.py` — backup categorizer
- `src/prepare_data.py` — data preparation
- `src/inspect_data.py` — data inspection
- `src/check_body_integrity.py` — body integrity checks
- `src/create_labeling_queue.py` — labeling queue creation
- `src/create_validation_sample.py` — validation sample creation
- `src/sample_emails.py` — email sampling utility
- `check_accuracy.py` — validation accuracy checker
- `update_verified_labels.py` — verified-label update utility
- `data/` — project data
- `models/` — model-related files
- `outputs/` — generated CSV files
- `requirements.txt` — Python dependencies

## Setup
Open PowerShell in the project directory and activate the virtual
environment:

    .\.venv\Scripts\Activate.ps1

Install dependencies if needed:

    python -m pip install -r requirements.txt

## Run the Project
Run the categorizer:

    python .\src\auto_categorize.py

Check validation accuracy:

    python .\check_accuracy.py

## Generated Outputs
- `outputs/auto_categorized_emails.csv`
- `outputs/emails_needing_review.csv`

Other output files support data preparation, labeling, and validation.

## Validation Results
The latest reported validation result is 42 correct predictions out of
42 manually verified emails (100% accuracy).

This result applies only to the verified sample and does not guarantee
the same accuracy on all emails.

## Limitations
The system relies on predefined classification rules. Emails with
ambiguous wording or unfamiliar patterns may require manual review.
