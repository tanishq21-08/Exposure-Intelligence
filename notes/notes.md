On my first extraction:
1) Unit conversion was wrong because the prompt can't tell which sheet uses which unit (design bug)
2) verbatim/derived labels are wrong because your definitions aren't sharp enough( prompt bug)
3) Confidence is uniformly 1 and untrustworthy including on wrong values

Made some changes in the prompt:
1) Unit conversion handled well
2) Derived or verbatim labels are now strong enough
3) Confidence is still not trustworthy because the in case of broker A for concrete the construction became reinforced concrete with a confidence score of 0.8 and for the broker B for concrete the construction was reinforced concrete again but with a confidence score of 1.0

Next changes to tackle point 3:

Now added a confidence layer.I will ask the LLM to extract not once but N number of times and then i will assign the value that has been fetched the maximum number of times (majority vote) from all the values fetched and then I will add a confidence score based on the percentage and not asking the LLM itself to grade it.

After these changes what I observed was this:
For every genuinely non ambigious field, the confidence is 1.0 that is fine but when it is ambiguous, it gives me different values and also different levels of confidence, which is worth noting and this tells us that it would be better for the underwriter to manually go through the fields where there is an ambiguity

for example: occupancy "Light Industrial" gave me 3 different answers across 5 runs 0.4 confidence-> correctly flagged as unresolvable

Limitation:confidence numbers are noisy at n=5 (same field ranged 0.4-0.8 across runs);
more samples stabilizes but costs more


