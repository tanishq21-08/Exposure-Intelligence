On my first extraction:
1) Unit conversion was wrong because the prompt can't tell which sheet uses which unit (design bug)
2) verbatim/derived labels are wrong because your definitions aren't sharp enough( prompt bug)
3) Confidence is uniformly 1 and untrustworthy including on wrong values

Made some changes in the prompt:
1) Unit conversion handled well
2) Derived or verbatim labels are now strong enough
3) Confidence is still not trustworthy because the in case of broker A for concrete the construction became reinforced concrete with a confidence score of 0.8 and for the broker B for concrete the construction was reinforced concrete again but with a confidence score of 1.0
