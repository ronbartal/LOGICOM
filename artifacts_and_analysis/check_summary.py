import json

with open('LOGICOM_Data_Analysis_clean.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Get M_F section from cell 97
source = ''.join(nb['cells'][97].get('source', []))

# Find M_F Case Deep Dive section
start = source.find('**M_F Case Deep Dive**')
end = source.find('#### Important Caveats')

if start > 0 and end > start:
    mf_section = source[start:end]
    
    # Save to file
    with open('mf_section.txt', 'w', encoding='utf-8') as out:
        out.write(mf_section)
    print("M_F section saved to mf_section.txt")
