import numpy as np

locations = [
'/location/neighborhood/neighborhood_of',
'/location/fr_region/capital',
'/location/cn_province/capital',
'/location/in_state/administrative_capital',
'/base/locations/countries/states_provinces_within',
'/people/person/place_of_birth',
'/people/deceased_person/place_of_death',
'/location/it_region/capital',
'/location/us_state/capital',
'/location/us_county/county_seat',
'/location/in_state/legislative_capital',
'/sports/sports_team/location',
'/location/in_state/judicial_capital',
'/people/family/country',
'/time/event/locations',
'/business/company/place_founded',
'/location/administrative_division/country',
'/location/br_state/capital',
'/location/mx_state/capital',
'/location/province/capital',
'/people/deceased_person/place_of_burial',
'/people/person/place_lived',
'/broadcast/producer/location',
'/broadcast/content/location',
'/location/jp_prefecture/capital',
'/film/film/featured_film_locations',
'/people/place_of_interment/interred_here',
'/location/de_state/capital',
'/business/company/locations',
'/location/country/capital',
'/location/location/contains',
'/film/film_festival/location'
]

people_companies = [
'/business/company/founders',
'/people/family/members',
'/people/profession/people_with_this_profession',
'/business/company/advisors',
'/business/shopping_center/owner',
'/people/person/children',
'/business/company_advisor/companies_advised',
'/business/person/company',
'/business/company/major_shareholders',
'/business/business_location/parent_company'
]

misc = [
'NA',
'/location/country/languages_spoken',
'/people/person/religion',
'/people/ethnicity/included_in_group',
'/people/person/nationality',
'/business/shopping_center_owner/shopping_centers_owned',
'/people/person/ethnicity',
'/people/ethnicity/geographic_distribution',
'/people/person/profession',
'/location/country/administrative_divisions',
'/film/film_location/featured_in_films',
'/people/ethnicity/includes_groups'
]


def assign_indices(data):
    """
    Replaces all the labels in the sub-category lists by the index that was assigned to them when preprocessing
    """
    global locations
    global people_companies
    global misc

    for i in xrange(0, len(locations)):
        locations[i] = data._rel_dict[locations[i]]
    for i in xrange(0, len(people_companies)):
        people_companies[i] = data._rel_dict[people_companies[i]]
    for i in xrange(0, len(misc)):
        misc[i] = data._rel_dict[misc[i]]

    locations = np.array(locations)
    people_companies = np.array(people_companies)
    misc = np.array(misc)

def negations_for_label(label):
    """
    Returns all the labels that negate the given label
    Only the worst-offender labels are returned, i.e. labels that belong to another sub-category
    """

    global locations
    global people_companies
    global misc

    if label in locations:
        negate = np.hstack((people_companies, misc))
    elif label in people_companies:
        negate = np.hstack((locations, misc))
    else:
        negate = np.hstack((locations, people_companies))

    return negate