PYTHON ?= /Users/yaminarefeen/anaconda3/bin/python

RESULTS := results
LONG     := $(RESULTS)/long_data.parquet
SELECT   := $(RESULTS)/ours_selection.csv
CLAIM1_T := $(RESULTS)/claim1_table.csv
CLAIM2_T := $(RESULTS)/claim2_table.csv
CLAIM1_M := $(RESULTS)/claim1_summary.md
CLAIM2_M := $(RESULTS)/claim2_summary.md

.PHONY: all clean load select test summarize

all: $(CLAIM1_M) $(CLAIM2_M)

$(LONG): src/load_data.py config.yaml $(wildcard data/*.csv)
	$(PYTHON) src/load_data.py

$(SELECT): src/select_ours.py $(LONG) config.yaml
	$(PYTHON) src/select_ours.py

# Both claim tables are written by a single invocation; declare the second
# as depending on the first so the script runs exactly once on GNU Make 3.81.
$(CLAIM1_T): src/run_tests.py $(LONG) $(SELECT) config.yaml
	$(PYTHON) src/run_tests.py

$(CLAIM2_T): $(CLAIM1_T)

$(CLAIM1_M): src/make_summaries.py $(CLAIM1_T) $(CLAIM2_T) config.yaml
	$(PYTHON) src/make_summaries.py

$(CLAIM2_M): $(CLAIM1_M)

load:      $(LONG)
select:    $(SELECT)
test:      $(CLAIM1_T) $(CLAIM2_T)
summarize: $(CLAIM1_M) $(CLAIM2_M)

clean:
	rm -rf $(RESULTS)
