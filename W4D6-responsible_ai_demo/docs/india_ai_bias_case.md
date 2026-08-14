# India AI Bias Case Analysis — Facial Recognition

## Case

Facial Recognition Technology (FRT) is increasingly being considered and deployed in India for surveillance, security and identification. A 2025 study in the *Journal of Development Policy and Practice* highlights concerns that facial-recognition systems can produce biased outcomes, particularly against women and minority communities. This is important because incorrect identification can have serious consequences when such systems are used in policing or public surveillance.

## Root Cause

The bias may not come from the algorithm alone. It can result from imbalanced training and reference datasets, differences in image quality, insufficient representation of demographic groups, and deployment conditions that differ from those used during development. When some groups are underrepresented, model performance can vary across demographic groups.

## Proposed Fix

Before deployment, systems should be evaluated separately across demographic groups. Developers should use more representative datasets, measure false-positive and false-negative rates for different groups, conduct independent bias audits, and continuously monitor performance after deployment. Human review should also remain part of high-stakes identification decisions.

## Responsible AI Lesson

This case shows why overall accuracy is not enough. AI systems should be evaluated for subgroup fairness, documented transparently, and monitored throughout their lifecycle. Responsible AI requires considering who benefits, who may be harmed, and whether model performance is consistent across different populations.

## Source

Basheer, I. P. (2025). "Bias in the Algorithm: Issues Raised Due to Use of Facial Recognition in India." *Journal of Development Policy and Practice*, 10(1), 61–79.