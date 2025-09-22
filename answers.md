# Short Answers

**1. Improving with 200 Labels**  
Apply **data augmentation** (e.g., synonym replacement, back-translation) and **semi-supervised learning** (pseudo-labeling on unlabeled data) to expand effective training samples without massive manual labeling.

**2. Mitigating Bias & Unsafe Outputs**  
Conduct **bias audits** on model predictions using a demographically balanced validation set, implement **human-in-the-loop** review on uncertain cases, and enforce rule-based filters to block unsafe content.

**3. Prompt Design for Personalized Openers**  
Use **few-shot examples** that illustrate personalization, include **explicit instructions** for tone and context, and leverage **dynamic placeholders** (e.g., {{name}}, {{company}}) to ensure relevance and avoid generic phrasing.
