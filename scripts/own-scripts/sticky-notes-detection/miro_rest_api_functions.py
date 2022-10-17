# pip install requests

from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
import requests
import asyncio
import nest_asyncio
nest_asyncio.apply()

# board_name = "STICKY NOTES SYNC BOARD"
# board_description = "Board to test the synchronization of the sticky notes in the real life and the here created miro board."


# VARS FOR MIRO REST API
DEBUG_PRINT_RESPONSES = False
auth_token = "eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_Bw1UPxNElmWbBGy8MfSWWJWOCLs"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {auth_token}"
}


# MIRO REST API HELPER FUNCTIONS
# GET ALL BOARD IDS AND NAMES
# WARNING:
# seems that the Get boards REST API function is not working properly
# sometimes it only returns one board, even if there are more


async def get_all_miro_board_names_and_ids(session):
    url = "https://api.miro.com/v2/boards?limit=50&sort=default"
    async with session.get(url, headers=headers) as resp:
        response = await resp.json()

        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())
        return [{
            "name": board['name'],
            "id": board['id']
        } for board in response['data']]


# GET ALL BOARD ITEMS
# default limit is maximum (50)
async def get_all_items(item_type: str, max_num_of_items: int = 50):
    global global_session
    global global_board_id

    url = f"https://api.miro.com/v2/boards/{global_board_id.replace('=', '%3D' )}/items?limit={max_num_of_items}&type={item_type}"

    async with global_session.get(url, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())
        return response['data']


# DELETE BOARD ITEM
async def delete_item(item_id: str):
    global global_board_id
    global global_session
    url: str = f"https://api.miro.com/v2/boards/{global_board_id.replace('=', '%3D')}/items/{item_id}"

    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_Bw1UPxNElmWbBGy8MfSWWJWOCLs"
    }

    response = requests.delete(url, headers=headers)
    if DEBUG_PRINT_RESPONSES:
        print(await response.text())

    # WARNING: Seems not to work with global_session.delete(url, headers=headers)
    #     async with global_session.delete(url, headers=headers) as resp:
    #         response = await resp.json()
    #         if DEBUG_PRINT_RESPONSES: print(await resp.text())


# DELETE ALL BOARD ITEMS
async def delete_all_items(item_type: str):
    global global_session
    board_items = await asyncio.create_task(get_all_items(item_type))
    for board_item in board_items:
        await asyncio.create_task(delete_item(board_item['id']))


# CREATE FRAME
async def create_frame(
    frame_x: int,
    frame_y: int,
    frame_text: str,
    frame_height: int,
    frame_width: int,
    board_id: str,
    session
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/frames"
    payload = {
        "data": {
            "format": "custom",
            "title": frame_text,
            "type": "freeform"
        },
        "style": {"fillColor": "#ffffffff"},
        "position": {
            "origin": "center",
            "x": frame_x,
            "y": frame_y
        },
        "geometry": {
            "height": frame_height,
            "width": frame_width
        }
    }

    headers = {
        "accept": "*/*",
        "content-type": "application/json",
        "authorization": f"Bearer {auth_token}"
    }

    async with session.post(url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())


# CREATE ITEM
async def create_item(
    sticky_note_position,
    color: str,
    sticky_note_text: str,
    board_id: str,
    session
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/sticky_notes"
    payload = {
        "data": {
            "content": sticky_note_text,
            "shape": "square"
        },
        "style": {"fillColor": color},
        "position": {
            "origin": "center",
            "x": sticky_note_position['xmin'],
            "y": sticky_note_position['ymin']
        },
        "geometry": {
            #             "height": sticky_note_position['ymax'] - sticky_note_position['ymin'],
            "width": sticky_note_position['xmax'] - sticky_note_position['xmin']
        }
    }

    async with session.post(url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())


async def create_image(
    image_position,
    image_text: str,
    board_id,
    session
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/images"
    payload = {
        "title": image_text,
        "position": {
            "origin": "center",
            "x": image_position['xmin'],
            "y": image_position['ymin']
        },
        "geometry": {
            #             "height": sticky_note_position['ymax'] - sticky_note_position['ymin'],
            "height": image_position['ymax'] - image_position['ymin'],
            "width": image_position['xmax'] - image_position['xmin']
        }
    }

    async with session.post(url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

# CREATE LIST OF ITEMS


async def create_all_items(sticky_note_positions):
    global global_session
    global global_board_id

    url = f"https://api.miro.com/v2/boards/{global_board_id.replace('=', '%3D')}/sticky_notes"

    for sticky_note_position in sticky_note_positions:
        payload = {
            "data": {"shape": "square"},
            "position": {
                "origin": "center",
                "x": sticky_note_position['xmin'],
                "y": sticky_note_position['ymin']
            },
            "geometry": {
                #             "height": sticky_note_position['ymax'] - sticky_note_position['ymin'],
                "width": sticky_note_position['xmax'] - sticky_note_position['xmin']
            }
        }

        async with global_session.post(url, json=payload, headers=headers) as resp:
            response = await resp.json()
            if DEBUG_PRINT_RESPONSES:
                print(await resp.text())


# CREATE NEW MIRO-BOARD
# (if no Board with given name exists, else return the id or the existing one)
async def create_new_miro_board_or_get_existing(name: str, description: str, session):
    board_names_and_ids = await asyncio.create_task(get_all_miro_board_names_and_ids(session))

    for board_name_and_id in board_names_and_ids:
        if board_name_and_id['name'] == name:
            print(f"WARNING: The board with the name {name} already exist. \n")
            return board_name_and_id['id']

    url = "https://api.miro.com/v2/boards"

    payload = {
        "name": name,
        "description": description,
        "policy": {
            "permissionsPolicy": {
                "collaborationToolsStartAccess": "all_editors",
                "copyAccess": "anyone",
                "sharingAccess": "team_members_with_editing_rights"
            },
            "sharingPolicy": {
                "access": "private",
                "inviteToAccountAndBoardLinkAccess": "no_access",
                "organizationAccess": "private",
                "teamAccess": "private"
            }
        }
    }

    async with session.post(url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

    return response['id']
