import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Beer', 'Bread'],
    ['Milk', 'Bread', 'Beer', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Beer'],
    ['Milk', 'Bread'],
    ['Beer', 'Chips'],
    ['Milk', 'Bread', 'Chips'],
    ['Bread', 'Butter', 'Chips'],
    ['Milk', 'Beer', 'Bread']
]

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

if frequent_itemsets.empty:
    print("No frequent itemsets found. Try lowering min_support.")
else:
    print("=== Frequent Itemsets ===")
    print(frequent_itemsets.sort_values(by='support', ascending=False))

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

if rules.empty:
    print("\nNo association rules found. Try lowering the confidence threshold.")
else:
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    print("\n=== Association Rules ===")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
