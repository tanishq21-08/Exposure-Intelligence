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

Now calibration whether the confidence layer actually works:

I added the calibration layers and that gave me 82% accuracy for the areas where LLM was 100% confident. So clearly the model is overconfident and I measured the gap is ~0.18 ECE

1.0-confidence fields were only 82% accurate->ECE 0.189->self-consistency confidence is overconfident

Root Cause: self-consistency catches  wavering, not consistent-but-wrong errors(the coherent-but-wrong-gap)

Caveats: n=36(noisy), strict grading of ambigious fields inflates error somewhat

# Two next steps could be:
1) Fix the overconfidence by tempearture scaling calibration (Guo's paper):Apply a correction that scales the confidence numbers down so 1.0-claims become ~0.82, making the scores honest.This closes the loop fully-"I detecetd the overconfidence and corrected it, dropping the ECE from 0.189 to X."That's the complete, impressive version.
2) Run it on more data to get a less noisy ECE.



After adding the calibration layer, I modularized the code and added the config file

Now adding resilience layer...why? because if my api call fails on let's say 5th attempt, I don't want that my credits and money on my 4th attempt are also wasted, so I add a resilience layer like if the api call fails, wait for a while and then retry and then if again it fails, wait for some more time and then again retry...( learn more aboout this in detail from tutorials, how API calls actually work internally)

Next step is Caching: Why Caching?

If I am giving the same input twice, so it won't recall APIs, it won't call the LLM twice with the same inputs. It will use the already used result.

( However, this step is just an engineering step and doesn't suit here well nor is recommended becasue with temperature being 0,7, we might expect different outcomes for the same input)