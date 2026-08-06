import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from schema import Portfolio          # <-- importing from your own module
from config import config

load_dotenv()
client = OpenAI()

logging.basicConfig(level=logging.INFO)   # so we can see retry messages
logger = logging.getLogger(__name__)

SYSTEM = """You extract commercial-property exposure data from messy broker Statements of Value.You are expert in this field and you realise that even a small error and cause a magnificent loss to the insurance companies so you are super careful while you fill in the values.

Return one entry per property location. For EVERY field, provide four things:
- value: the normalized answer
- confidence: your certainty from 0.0 to 1.0
- source: the exact text from the document you used
- type: "verbatim" if copied directly from the text, "derived" if you inferred, converted, or guessed it

Rules:
- construction must be one of: Steel Frame, Reinforced Concrete, Masonry, Timber Frame, UNKNOWN
- occupancy must be one of: Warehouse/Storage, Office, Retail, Restaurant, Manufacturing, Mixed/Other, UNKNOWN
- sprinklered must be one of: Y, N, UNKNOWN
- Convert values to standard form: '£2.4m' -> 2400000 (pure numbers). Area must end up in square feet: read the ORIGINAL unit from the header/context; if already sq ft keep it (verbatim); if in another unit convert to sq ft and mark derived, keeping the original in source.
- type: "verbatim" ONLY if character-for-character identical to source. Any typo fix, vocab mapping, or conversion is "derived".
- For the address: if clearly incomplete or references elsewhere ("see broker note", "TBC"), return what's there but set confidence low (~0.3) and type derived.
- If a value is genuinely missing, set value to null (numbers) or "UNKNOWN", and lower confidence. NEVER invent a value."""





def extract(text, temperature=None):
    if temperature is None:
        temperature = config["extraction_temperature"]

    max_retries = 3
    base_delay = 1          # seconds

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.beta.chat.completions.parse(
                model=config["model"],
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Statement of Values:\n\n{text}"},
                ],
                response_format=Portfolio,
            )
            return completion.choices[0].message.parsed          # success -> return immediately

        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Extraction failed after {max_retries} attempts: {e}")
                raise                                            # give up, re-raise the error
            delay = base_delay * (2 ** (attempt - 1))            # exponential backoff: 1, 2, 4...
            logger.warning(f"Attempt {attempt} failed ({e}); retrying in {delay}s...")
            time.sleep(delay)