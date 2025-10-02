# after reading Excel with load_excel(...)
emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_deductions = prepare_inputs(load_excel_bytes)

interim = build_interim(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year=2025, pay_deductions=pay_deductions)
final   = build_final(interim)
penalty = build_penalty_dashboard(interim)

# save all 3 tabs in one workbook
data = save_excel_outputs(interim, final, 2025, penalty_dashboard=penalty)
with open("Final_Interim_Penalty_2025.xlsx","wb") as f:
    f.write(data)
