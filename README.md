The Statement of Values given to the insurance companies from different brokers come in different formats, with different headings, columns etc. The data needs an initial processing and also there isn't a fixed format that the different brokers or clientd follow, it can be structured, messy, in form of an excel file, pdf, handwritten docs etc., there isn't a universal structure or format that is followed by this companies. Now I need the most important attributes for a property, like the type of construction, the year it was built in, insured value etc. and I need to convert this messy and unstructured data from different brokers into a structured format.
The desired schema for my design is:
1)Source 2)Ref 3) Normalized Address 4)TIV 5)Construction 6)Occupancy 7)Year Built 8)Floor Area 9)Sprinkled 10)Ambiguity/Note
And each field will have a metadata to know if it can be trusted, so each field value will have 4 sub-fields:1)Value 2)confidence 3)source 4)type

Examples of the mess that can be seen in the date from two different sources:
Location v/s Property Address, Sum Insured/ TIV (Total Value Insured), Missing values, abbreviated values for example 2.4m pounds v/s 2400000, requirement to read broker's note etc.
