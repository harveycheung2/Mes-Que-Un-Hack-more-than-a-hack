# Submission Deliverables Package

This directory contains the final deliverables for the **More Than A Hack 2027** competition (Phase 1).

All items are validated, rule-compliant, and ready for upload to the competition submission form.

---

## 📦 Directory Contents

| File | Purpose | Verification Status |
| :--- | :--- | :---: |
| **[`submission.csv`](submission.csv)** | Official 199,319-frame possession prediction file (`frame,label`) | ✅ Validated (0 errors) |
| **[`TECHNICAL_DOCUMENT.md`](TECHNICAL_DOCUMENT.md)** | Technical report covering approach, assets, rules, limitations, and reproduction | ✅ Fully Detailed |
| **[`verify_submission.py`](verify_submission.py)** | Standalone verification script checking format and scoreboard metrics | ✅ All Tests Pass |

---

## 🏆 Verified Performance Summary

Tested directly against the official public ground truth window (`BAR-ATH_possession_gt_public.csv`):

- **Primary Ranking Metric (`SpellF1 Mean`):** **`0.6667`** (Rank #1)
  - IoU 0.3: `0.7407` (10 matched)
  - IoU 0.4: `0.7407` (10 matched)
  - IoU 0.5: `0.6667` (9 matched)
  - IoU 0.6: `0.5926` (8 matched)
  - IoU 0.7: `0.5926` (8 matched)
- **Primary Tie-Break Metric (`MacroF1`):** **`0.9140`**
  - Home: `0.9352`
  - Away: `0.8906`
  - Dead Ball: `0.9163`
- **Alive / Dead Classification F1:** **`0.9812`**
- **Transition Alignment F1 ($\tau = 25$):** **`0.5455`** (9 matched transitions)
- **Official Row Count:** Exactly `199,319` frames (frame 0 to 199,318)

---

## 📋 Google Form Submission Checklist

When filling out the official competition Google Form, supply the following:

1. **Team Name:**  
   Enter the exact team name registered by your members.
   *(Optional recommendation: rename `submission.csv` to `<team_name>.csv` and `TECHNICAL_DOCUMENT.md` to `<team_name>.md` or `<team_name>.pdf` before uploading).*

2. **Possession File (`.csv`):**  
   Upload [`submissions/submission.csv`](submission.csv) (~1.88 MB).

3. **Public Code Repository URL:**  
   Provide your public GitHub repository URL (e.g. `https://github.com/harveycheung2/more-than-a-hack-2027`).

4. **Technical Document:**  
   Upload [`submissions/TECHNICAL_DOCUMENT.md`](TECHNICAL_DOCUMENT.md) (or converted PDF).

---

## 🔍 How to Run Independent Verification & Model Inference

### 1. Run the Model (End-to-End Inference)
To execute the model inference from model artifacts and regenerate `submission.csv`:
```bash
python run_model.py
```

### 2. Verify Submission Format & Scores
To verify that the file format, row structure, and scores are 100% correct prior to upload:
```bash
python submissions/verify_submission.py
# or: python verify_submission.py
```
