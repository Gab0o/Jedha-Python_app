def parse_box_office(raw_box_office):
    if(raw_box_office != 'N/A'):
        return raw_box_office.replace('$', '').replace(',', '')
    else:
        return None
