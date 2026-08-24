# Retrieval Ablation

Rank of the expected source document across 16 answerable questions. Lower is better; `-` means the document was never retrieved.


## Summary

| Configuration | Hit@1 | Hit@3 | Hit@4 | MRR | Never retrieved |
|---|---|---|---|---|---|
| BM25 only | 62% | 100% | 100% | 0.79 | 0 |
| Dense only | 94% | 100% | 100% | 0.96 | 0 |
| Hybrid (RRF) | 94% | 100% | 100% | 0.97 | 0 |
| Hybrid + rerank | 100% | 100% | 100% | 1.00 | 0 |

## Chunk-level — rank of the passage containing the answer

Document-level Hit@k is easy on a 7-document corpus. This asks the harder question: did the right *passage* outrank the other passages in the same file? (12 questions with a known expected value.)

| Configuration | Hit@1 | Hit@3 | Hit@4 | MRR | Never retrieved |
|---|---|---|---|---|---|
| BM25 only | 50% | 100% | 100% | 0.71 | 0 |
| Dense only | 92% | 100% | 100% | 0.94 | 0 |
| Hybrid (RRF) | 75% | 100% | 100% | 0.88 | 0 |
| Hybrid + rerank | 100% | 100% | 100% | 1.00 | 0 |

## Per question (document-level rank)

| Question | Expected document | BM25 | Dense | Hybrid | +rerank |
|---|---|---|---|---|---|
| What storage does the Standard tier include? | Pricing_and_SLA.pdf | 1 | 1 | 1 | 1 |
| What does Standard Tier Access mean for employees? | Security_Policy.pdf | 1 | 1 | 1 | 1 |
| What is document SEC-POL-007? | Security_Policy.pdf | 3 | 1 | 1 | 1 |
| How long do I have to report a suspected data breach? | Security_Policy.pdf | 2 | 1 | 1 | 1 |
| A Standard account had 99.3% uptime this month. What service credit applies? | Pricing_and_SLA.pdf | 1 | 1 | 1 | 1 |
| Can I get a refund on an annual plan after 20 days? | Pricing_and_SLA.pdf | 3 | 1 | 2 | 1 |
| Who approves access to the Atman Cloud Console? | Onboarding_Guide.pdf | 1 | 3 | 1 | 1 |
| What should I do if the appliance LED is blinking red? | Product_Manual.pdf | 1 | 1 | 1 | 1 |
| How much PTO do full-time employees accrue? | Employee_Handbook.pdf | 1 | 1 | 1 | 1 |
| What happens if I exceed the rate limit on the API? | API_Reference.pdf | 2 | 1 | 1 | 1 |
| How long is data kept after I cancel my subscription? | FAQ_Support.pdf | 1 | 1 | 1 | 1 |
| What happens at my Day 30 check-in? | Onboarding_Guide.pdf | 2 | 1 | 1 | 1 |
| How much storage does the CSP-400 have? | Product_Manual.pdf | 1 | 1 | 1 | 1 |
| Is two-factor authentication mandatory? | FAQ_Support.pdf | 1 | 1 | 1 | 1 |
| What does the company match on retirement contributions? | Employee_Handbook.pdf | 1 | 1 | 1 | 1 |
| How much PTO do I accrue? | Employee_Handbook.pdf | 2 | 1 | 1 | 1 |
