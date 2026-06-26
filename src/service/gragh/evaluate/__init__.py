"""
Citation-edge relation evaluation.

Build the local directed citation graph (paper i cites paper j), then ask an LLM
to judge the relation type of each directed edge from its in-text citation
contexts. The current label set is ``inheritance`` (knowledge inheritance) and
``unknown`` (cannot be determined).
"""
