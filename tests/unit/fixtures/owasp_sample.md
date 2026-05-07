## LLM01: Prompt Injection

Prompt injection vulnerabilities occur when user input alters the LLM's intended
behavior through manipulated prompts. Direct prompt injection overwrites system
prompts, while indirect injection exploits external inputs from files or websites.

### Example Attack

An attacker crafts an input that instructs the model to ignore previous instructions
and reveal the system prompt, or to perform actions outside its intended scope.

### Mitigation

Enforce privilege control and apply human oversight for high-impact actions.
Separate system prompts from user input to prevent cross-boundary manipulation.
