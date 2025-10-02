# 1) Guarantee 12 months per employee
print(set(interim.groupby("employeeid")["monthnum"].nunique().tolist()))
# Expect: {12}

# 2) See how many months are FT vs PT
print(interim["ft"].value_counts(dropna=False))
print(interim["parttime"].value_counts(dropna=False))

# 3) Inspect the raw status tokens we detected
st = emp_status.copy()
if not st.empty:
    st["_estatus_norm"] = st["employmentstatus"].astype(str).map(lambda s: re.sub(r"[^A-Z0-9]","", s.upper()))
    st["_role_norm"] = st.get("role", pd.Series("", index=st.index)).astype(str).map(lambda s: re.sub(r"[^A-Z0-9]","", s.upper()))
    print("employmentstatus tokens:", st["_estatus_norm"].value_counts().head(20))
    print("role tokens:", st["_role_norm"].value_counts().head(20))
