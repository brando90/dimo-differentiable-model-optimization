"""
For making a mini-Places dataset in the same organization as miniImageNet
The paper proposes the mini_imagenet - ;

But perhaps it is better to just read how the mini-imagenet data set was created and create the mini-diversity data set
using *that* format.
- do we really need them to be same size?
- smaller images are good for quicker experiments, especially because episodic meta-learning is expensive.
"""

import os
import shutil
import numpy as np
from random import sample

# the path to the dataset folder ... 
dataset_path = '/shared/rsaas/m2yuan/data/vision/torralba/deeplearning/images256/'  # there are some
mini_path = '/home/m2yuan/mini_places/'


def get_all_catgories(dataset_path):
    full_list = []
    all_letters = os.listdir(dataset_path)
    for letter in all_letters:
        letter_path = dataset_path + letter
        cat_names = os.listdir(letter_path)
        full_list = full_list + cat_names
    for cat in full_list:
        num = len(os.listdir(os.path.join(dataset_path, cat[0], cat)))
        if num <= 600:
            full_list.remove(cat)

    print(full_list)  # check the full_list by printing
    print("there are {} categories in total.".format(len(full_list)))
    cat_list = sample(full_list, 100)
    print(cat_list)  # check the full_list by printing
    print("there are {} categories in cat_list.".format(len(cat_list)))
    return cat_list


# run sampling on all categories, divide them into three sets (training/validation/test)
def create_divided_set(cat_list, set_name, dataset_path=dataset_path, mini_path=mini_path):
    if set_name == 'train':
        cat_list = cat_list[0:64]
    if set_name == 'val':
        cat_list = cat_list[64:80]
    if set_name == 'test':
        cat_list = cat_list[80:100]

    print("cat_list {} of {} set is :\n".format(cat_list, set_name))
    mini_set_path = os.path.join(mini_path + set_name)
    os.mkdir(mini_set_path)

    for cat in cat_list:
        load_cat_path = os.path.join(dataset_path, cat[0], cat)
        save_cat_dir = os.path.join(mini_set_path, cat)  # directly create a category with its name
        os.mkdir(save_cat_dir)
        all_items = os.listdir(load_cat_path)
        try:
            # why sometimes, the following lines are broken ... ??? 
            item_list = sample(all_items, 600)
            for item in item_list:
                shutil.copyfile(load_cat_path + '/' + item, save_cat_dir + '/' + item)
        except:
            print("copy failure in cat: [{}]".format(cat))


def make_up_failed_cat(cat, set_name, dataset_path=dataset_path, mini_path=mini_path):
    # there are some 
    load_cat_path = os.path.join(dataset_path, cat[0], cat)
    save_cat_path = os.path.join(mini_path, set_name, cat)
    # os.system( 'rm -r {}'.format(save_cat_path) )
    os.mkdir(save_cat_path)
    all_items = os.listdir(load_cat_path, )
    try:
        item_list = sample(all_items, 600)
        for item in item_list:
            shutil.copyfile(load_cat_path + '/' + item, save_cat_path + '/' + item)
    except:
        print("copy failed in cat: [{}]".format(cat))


def main():
    # cat_list = get_all_catgories(dataset_path) # for the first time ... 
    cat_list = ['aquarium', 'construction_site', 'windmill', 'mausoleum', 'ice_skating_rink', 'fire_escape',
                'herb_garden', 'bedroom', 'game_room', 'rainforest', 'boat_deck', 'water_tower', 'office_building',
                'sea_cliff', 'skyscraper', 'cottage_garden', 'ocean', 'botanical_garden', 'hospital', 'art_studio',
                'ice_cream_parlor', 'dam', 'shoe_shop', 'bayou', 'kitchen', 'arch', 'highway', 'valley', 'pavilion',
                'classroom', 'abbey', 'ballroom', 'schoolhouse', 'snowfield', 'auditorium', 'plaza', 'iceberg',
                'bowling_alley', 'building_facade', 'bridge', 'home_office', 'beauty_salon', 'dorm_room', 'viaduct',
                'courthouse', 'phone_booth', 'raft', 'shopfront', 'boardwalk', 'closet', 'galley', 'campsite',
                'pasture', 'wind_farm', 'hospital_room', 'television_studio', 'yard', 'aqueduct', 'airport_terminal',
                'cockpit', 'attic', 'engine_room', 'mansion', 'conference_center', 'hotel_room', 'gift_shop', 'pantry',
                'restaurant_kitchen', 'golf_course', 'pulpit', 'kasbah', 'boxing_ring', 'chalet', 'creek',
                'topiary_garden', 'excavation', 'train_railway', 'pagoda', 'veranda', 'restaurant', 'crosswalk',
                'laundromat', 'driveway', 'lighthouse', 'ski_slope', 'volcano', 'corn_field', 'fairway', 'motel',
                'basilica', 'butchers_shop', 'rope_bridge', 'pond', 'palace', 'slum', 'corridor', 'swamp',
                'forest_road', 'conference_room', 'badlands']
    set_names = ['val', 'test']
    for set_name in set_names:
        create_divided_set(cat_list, set_name)
    # for later making up category data ..
    for cat in cat_list:
        full_list.remove(cat)
    print("the left categories are: \n{}".format(full_list))


if __name__ == '__main__':
    full_list = ['rope_bridge', 'rainforest', 'ruin', 'racecourse', 'restaurant_kitchen', 'rice_paddy',
                 'restaurant_patio', 'railroad_track', 'reception', 'raft', 'river', 'residential_neighborhood',
                 'restaurant', 'runway', 'gas_station', 'garbage_dump', 'galley', 'golf_course', 'gift_shop',
                 'game_room', 'yard', 'formal_garden', 'fountain', 'fire_station', 'food_court', 'fire_escape',
                 'fairway', 'forest_path', 'forest_road', 'patio', 'pagoda', 'pantry', 'parking_lot', 'palace',
                 'pasture', 'playground', 'plaza', 'picnic_area', 'phone_booth', 'parlor', 'pavilion', 'pond', 'pulpit',
                 'engine_room', 'excavation', 'wind_farm', 'water_tower', 'wheat_field', 'waiting_room', 'windmill',
                 'watering_hole', 'office_building', 'orchard', 'ocean', 'office', 'locker_room', 'lighthouse', 'lobby',
                 'living_room', 'laundromat', 'driveway', 'dining_room', 'dock', 'dorm_room', 'dam', 'volcano',
                 'veranda', 'viaduct', 'valley', 'vegetable_garden', 'nursery', 'kasbah', 'kindergarden_classroom',
                 'kitchenette', 'kitchen', 'creek', 'closet', 'crevasse', 'cottage_garden', 'corn_field', 'cockpit',
                 'castle', 'crosswalk', 'conference_center', 'candy_store', 'cemetery', 'campsite', 'canyon',
                 'courtyard', 'chalet', 'construction_site', 'clothing_store', 'coast', 'conference_room', 'classroom',
                 'cafeteria', 'coffee_shop', 'corridor', 'courthouse', 'marsh', 'music_studio', 'medina', 'mountain',
                 'mausoleum', 'museum', 'motel', 'martial_arts_gym', 'mountain_snowy', 'mansion', 'jail_cell',
                 'bamboo_forest', 'boardwalk', 'butte', 'basilica', 'boat_deck', 'bar', 'boxing_ring',
                 'botanical_garden', 'banquet_hall', 'bookstore', 'beauty_salon', 'baseball_field', 'badlands',
                 'building_facade', 'ballroom', 'bowling_alley', 'butchers_shop', 'bayou', 'bridge', 'bedroom',
                 'basement', 'bus_interior', 'tower', 'topiary_garden', 'tree_farm', 'television_studio',
                 'train_railway', 'trench', 'ice_skating_rink', 'iceberg', 'ice_cream_parlor', 'igloo', 'islet',
                 'amphitheater', 'aquarium', 'airport_terminal', 'amusement_park', 'assembly_line', 'abbey', 'aqueduct',
                 'art_studio', 'art_gallery', 'auditorium', 'arch', 'alley', 'attic', 'stadium', 'sea_cliff',
                 'snowfield', 'shower', 'sky', 'ski_resort', 'shed', 'shoe_shop', 'sandbar', 'skyscraper', 'swamp',
                 'ski_slope', 'staircase', 'schoolhouse', 'shopfront', 'slum', 'supermarket', 'harbor', 'hot_spring',
                 'hospital', 'home_office', 'herb_garden', 'hotel_room', 'hospital_room', 'highway']
    cat_list = ['aquarium', 'construction_site', 'windmill', 'mausoleum', 'ice_skating_rink', 'fire_escape',
                'herb_garden', 'bedroom', 'game_room', 'rainforest', 'boat_deck', 'water_tower', 'office_building',
                'sea_cliff', 'skyscraper', 'cottage_garden', 'ocean', 'botanical_garden', 'hospital', 'art_studio',
                'ice_cream_parlor', 'dam', 'shoe_shop', 'bayou', 'kitchen', 'arch', 'highway', 'valley', 'pavilion',
                'classroom', 'abbey', 'ballroom', 'schoolhouse', 'snowfield', 'auditorium', 'plaza', 'iceberg',
                'bowling_alley', 'building_facade', 'bridge', 'home_office', 'beauty_salon', 'dorm_room', 'viaduct',
                'courthouse', 'phone_booth', 'raft', 'shopfront', 'boardwalk', 'closet', 'galley', 'campsite',
                'pasture', 'wind_farm', 'hospital_room', 'television_studio', 'yard', 'aqueduct', 'airport_terminal',
                'cockpit', 'attic', 'engine_room', 'mansion', 'conference_center', 'hotel_room', 'gift_shop', 'pantry',
                'restaurant_kitchen', 'golf_course', 'pulpit', 'kasbah', 'boxing_ring', 'chalet', 'creek',
                'topiary_garden', 'excavation', 'train_railway', 'pagoda', 'veranda', 'restaurant', 'crosswalk',
                'laundromat', 'driveway', 'lighthouse', 'ski_slope', 'volcano', 'corn_field', 'fairway', 'motel',
                'basilica', 'butchers_shop', 'rope_bridge', 'pond', 'palace', 'slum', 'corridor', 'swamp',
                'forest_road', 'conference_room', 'badlands']
    print("length of full_list is {}".format(len(full_list)))
    print("length of cat_list is {}".format(len(cat_list)))
    for cat in cat_list:
        full_list.remove(cat)
    print("length of left_list is {}".format(len(full_list)))
    print("the left categories are: \n{}".format(full_list))

    make_up_failed_cat('ski_resort', 'train')
