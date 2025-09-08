# GIDCL Error Correction Component

## Inputs and Outputs

- **Input:** A dirty relational table with identified error cells (e.g. via an error detector) and any auxiliary context (such as schema, example clean/dirty pairs, or a graph representation of the table).
- **Output:** A repaired table with corrected values for the flagged cells. Optionally, a set of explicit correction rules/functions generated during the process.

## Implicit Correction (LLM Fine-tuning and Inference)

- **Prepare training data:** Generate or collect dirty–clean cell pairs for each attribute (e.g. by sampling from cleaned tables or using synthetic errors). Use the function-generation step (below) to augment this data.
- **Fine-tune LLM:** Perform supervised fine-tuning on a local large language model (e.g. Vicuna, Mistral) using the collected dirty/clean pairs. Use LoRA or similar techniques to train the model efficiently. Include example prompts with few-shot input/output pairs in training (in-context learning) to help the model generalize.
- **Augment with RAG:** Build a retrieval component to fetch relevant context for each error cell (such as similar row examples or external data). At inference, include retrieved snippets or database values in the LLM prompt to guide correction.
- **Inference:** For each flagged cell, prompt the fine-tuned LLM with the current row’s context (other column values, schema hints, etc.) and ask it to generate the corrected value. The LLM implicitly “knows” the correction from its fine-tuning and in-context examples.

## Explicit Correction (Rule/Function Generation)

- **Generate candidate function:** Use the LLM in a code-generation or templated prompt mode to produce an *interpretable* correction function for the dirty cell. For example, prompt it to output a regex, arithmetic formula, or simple code snippet that transforms the given input to the correct output.
- **Apply and test:** Run the generated function on the dirty value to obtain a candidate repair. Verify this result against expected constraints (e.g., matches known clean values or fits type constraints).
- **Iterate if needed:** If the correction is incorrect or incomplete, feed back the result (and any discrepancy) to the LLM and ask it to refine the function. Repeat this loop until the function consistently produces the correct output for the example. This yields a human-readable rule capturing the error pattern.

## Selection Mechanism (Choosing Method per Cell)

- **Use error-detector signals:** For each cell flagged as erroneous, use the error detection model’s output or inferred error type to decide which strategy to apply.
- **Heuristic or learned choice:** For example, if the error is a formatting or pattern issue, prefer the explicit-function method; if it is semantic or context-dependent, use the implicit LLM generation. Alternatively, generate both candidates and pick the one with higher confidence or better consistency (e.g., the function output that passes more validity checks).
- **Per-cell decision:** In practice, each flagged cell is assigned to either the implicit or explicit correction path based on these criteria. This ensures the system flexibly uses the strengths of each approach rather than a one-size-fits-all fix.

## Graph-Based Refinement (Global Consistency via Learned Structure)

- **Graph construction:** Represent the table as a graph (e.g. nodes for columns or cells, edges for co-occurrence or similarity). Use a Graph Neural Network to learn structural patterns. GIDCL leverages a GNN on the relational graph to **capture global correlations and dependencies** among attributes[citedrive.com](https://www.citedrive.com/en/discovery/gidcl-a-graph-enhanced-interpretable-data-cleaning-framework-with-large-language-models/#:~:text=the challenges of traditional and,rectify complex dependencies and errors).
- **Discover dependencies:** From the learned graph structure, identify likely functional dependencies (FDs) or rules (e.g., “Attribute A uniquely determines B”). In effect, the system relearns which columns are inter-dependent based on the (partially cleaned) data.
- **Enforce and finalize:** Use the inferred dependencies to propagate and finalize repairs. For instance, if an FD A→B is detected, ensure that all rows sharing an A-value have consistent B-values (correcting any remaining mismatches). This global check catches residual errors and enforces consistency across the table. The final output satisfies both the local LLM-generated fixes and the global structure learned via the graph[citedrive.com](https://www.citedrive.com/en/discovery/gidcl-a-graph-enhanced-interpretable-data-cleaning-framework-with-large-language-models/#:~:text=the challenges of traditional and,rectify complex dependencies and errors).

Each component of this pipeline is implemented programmatically. The implicit LLM model is trained via fine-tuning (with ICL/RAG/SFT) to act as a predictor of the correct value. The explicit component uses the LLM as a rule generator that is iteratively refined. A simple decision logic (threshold or classifier) routes each error cell to the appropriate method. Finally, a GNN-based module relearns structural relationships to apply any remaining corrections for global consistency[citedrive.com](https://www.citedrive.com/en/discovery/gidcl-a-graph-enhanced-interpretable-data-cleaning-framework-with-large-language-models/#:~:text=the challenges of traditional and,rectify complex dependencies and errors)[citedrive.com](https://www.citedrive.com/en/discovery/gidcl-a-graph-enhanced-interpretable-data-cleaning-framework-with-large-language-models/#:~:text=The framework's creator,shot learning).

**Sources:** The above algorithmic flow follows the GIDCL framework, which combines LLM-based correction (implicit and explicit) with graph-structured refinement[citedrive.com](https://www.citedrive.com/en/discovery/gidcl-a-graph-enhanced-interpretable-data-cleaning-framework-with-large-language-models/#:~:text=the challenges of traditional and,rectify complex dependencies and errors)[citedrive.com](https://www.citedrive.com/en/discovery/gidcl-a-graph-enhanced-interpretable-data-cleaning-framework-with-large-language-models/#:~:text=The framework's creator,shot learning).