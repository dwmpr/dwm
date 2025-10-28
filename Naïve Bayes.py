import pandas as pd

data = {
    "CarNo":  [1,2,3,4,5,6,7,8,9,10],
    "Color":  ["Red","Red","Red","Yellow","Yellow","Yellow","Yellow","Yellow","Red","Red"],
    "Type":   ["Sports","Sports","Sports","Sports","Sports","SUV","SUV","SUV","SUV","Sports"],
    "Origin": ["Domestic","Domestic","Domestic","Domestic","Imported","Imported","Imported","Domestic","Imported","Imported"],
    "Stolen": ["Yes","No","Yes","No","Yes","No","Yes","No","No","Yes"]
}

df = pd.DataFrame(data)
class_counts = df["Stolen"].value_counts()
total_samples = len(df)
test = {"Color": "Red", "Type": "Sports", "Origin": "Domestic"}
results = {}

for c in class_counts.index:
    prior = class_counts[c] / total_samples
    prob = prior
    print(f"\nClass = {c}")
    print(f"  Prior P({c}) = {class_counts[c]}/{total_samples} = {prior:.3f}")
    for col in ["Color", "Type", "Origin"]:
        count = len(df[(df[col] == test[col]) & (df["Stolen"] == c)])
        cond_prob = count / class_counts[c] if class_counts[c] > 0 else 0
        prob *= cond_prob
        print(f"  Likelihood P({test[col]}|{c}) = {count}/{class_counts[c]} = {cond_prob:.3f}")
    results[c] = prob
    print(f"  Posterior (unnormalized) for {c} = {prob:.6f}")

predicted_class = max(results, key=results.get)
print("\nTest Sample:", test)
print("Predicted Class:", predicted_class)
