import pandas as pd
import itertools

# Read training data from CSV
data = pd.read_csv("training_data.csv")

print("Training Data:")
print(data)
print()


# ---------------------------------------------------
# Function to check whether a hypothesis covers example
# ---------------------------------------------------
def covers(hypothesis, example):

    for h, x in zip(hypothesis, example):

        # '?' means any value is accepted
        if h == '?':
            continue

        # Specific value must match
        if h != x:
            return False

    return True


# ---------------------------------------------------
# Generate all possible hypotheses
# ---------------------------------------------------
attributes = list(data.columns[:-1])
target = data.columns[-1]

domains = []

for attribute in attributes:
    values = list(data[attribute].unique())
    domains.append(values)


# Add '?' to every attribute domain
domains_with_question = []

for values in domains:
    domains_with_question.append(values + ['?'])


# Generate Cartesian product
all_hypotheses = list(
    itertools.product(*domains_with_question)
)

print("Total possible hypotheses:", len(all_hypotheses))
print()


# ---------------------------------------------------
# Find all hypotheses consistent with training data
# ---------------------------------------------------
consistent_hypotheses = []

for hypothesis in all_hypotheses:

    consistent = True

    for _, row in data.iterrows():

        example = row.iloc[:-1].tolist()
        actual_class = row.iloc[-1]

        prediction = covers(hypothesis, example)

        # Positive example must be covered
        if actual_class == 'Yes' and not prediction:
            consistent = False
            break

        # Negative example must NOT be covered
        if actual_class == 'No' and prediction:
            consistent = False
            break

    if consistent:
        consistent_hypotheses.append(hypothesis)


# ---------------------------------------------------
# Find Specific Boundary (S)
# ---------------------------------------------------
def is_more_specific(h1, h2):

    """
    Returns True if h1 is more specific than h2.
    """

    strictly_more_specific = False

    for a, b in zip(h1, h2):

        if a == b:
            continue

        # '?' is more general
        if b == '?':
            strictly_more_specific = True

        else:
            return False

    return strictly_more_specific


S = []

for h in consistent_hypotheses:

    is_most_specific = True

    for other in consistent_hypotheses:

        if h != other and is_more_specific(other, h):
            is_most_specific = False
            break

    if is_most_specific:
        S.append(h)


# ---------------------------------------------------
# Find General Boundary (G)
# ---------------------------------------------------
G = []

for h in consistent_hypotheses:

    is_most_general = True

    for other in consistent_hypotheses:

        if h != other and is_more_specific(h, other):
            is_most_general = False
            break

    if is_most_general:
        G.append(h)


# ---------------------------------------------------
# Display Results
# ---------------------------------------------------

print("==========================================")
print(" CANDIDATE-ELIMINATION RESULTS")
print("==========================================")

print("\nAttributes:")
print(attributes)

print("\nNumber of Consistent Hypotheses:")
print(len(consistent_hypotheses))


print("\nALL CONSISTENT HYPOTHESES (VERSION SPACE):")

for i, h in enumerate(consistent_hypotheses, 1):
    print(i, ":", h)


print("\nSPECIFIC BOUNDARY (S):")

for h in S:
    print(h)


print("\nGENERAL BOUNDARY (G):")

for h in G:
    print(h)
