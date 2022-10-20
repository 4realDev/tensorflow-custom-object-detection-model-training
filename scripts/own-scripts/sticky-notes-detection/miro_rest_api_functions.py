# pip install requests

import config
import nest_asyncio
import asyncio
import requests
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
from sklearn.feature_extraction import img_to_graph
from aiohttp import FormData

nest_asyncio.apply()


# board_name = "STICKY NOTES SYNC BOARD"
# board_description = "Board to test the synchronization of the sticky notes in the real life and the here created miro board."


# VARS FOR MIRO REST API
DEBUG_PRINT_RESPONSES = False
auth_token = "eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_N9OybOclP4WmwOKCNUjVuVMDshE"
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
    pos_x: int,
    pos_y: int,
    title: str,
    height: int,
    width: int,
    board_id: str,
    session
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/frames"
    payload = {
        "data": {
            "format": "custom",
            "title": title,
            "type": "freeform"
        },
        "style": {"fillColor": "#ffffffff"},
        "position": {
            "origin": "center",
            "x": pos_x,
            "y": pos_y
        },
        "geometry": {
            "height": height,
            "width": width
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

        return response['id']


# CREATE ITEM
async def create_item(
    pos_x,
    pos_y,
    width: int,
    color: str,
    text: str,
    board_id: str,
    parent_id,
    session
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/sticky_notes"
    payload = {
        "data": {
            "content": text,
            "shape": "square"
        },
        "style": {"fillColor": color},
        "position": {
            "origin": "center",
            "x": pos_x,
            "y": pos_y
        },
        "geometry": {
            "width": width
        },
        "parent": {"id": parent_id}
    }

    async with session.post(url=url, json=payload, headers=headers) as resp:
        response = await resp.json()
        # if DEBUG_PRINT_RESPONSES:
        print(await resp.text())


def create_line(
    pos_x,
    pos_y,
    width: int,
    height: int,
    color: str,
    board_id,
    parent_id,
    session
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/shapes"

    payload = {
        "data": {"shape": "round_rectangle"},
        "style": {"fillColor": color},
        "position": {
            "origin": "center",
            "x": pos_x,
            "y": pos_y
        },
        "geometry": {
            "height": height,
            "width": width
        },
        "parent": {"id": parent_id}
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.text)


async def create_image(
    pos_x,
    pos_y,
    width: int,
    title: str,
    path: str,
    board_id,
    parent_id,
    session
):
    url = f"https://api.miro.com/v2/boards/{board_id.replace('=', '%3D')}/images"

    headers = {
        "accept": "*/*",
        "authorization": f"Bearer {auth_token}"
    }

    payload = {
        "title": title,
        "position": {
            "x": pos_x,
            "y": pos_y,
            "origin": "center"
        },
        "geometry": {
            "width": width,
            "rotation": 0
        },
        "parent": {"id": parent_id}
    }

    # fmt: off
    data = FormData()
    data.add_field('resource', open(path, "rb"), filename=f'{title}.png', content_type="application/png")
    data.add_field('data', json.dumps(payload), content_type="application/json")
    # fmt: on

    async with session.post(url=url,  data=data, headers=headers) as resp:
        response = await resp.json()
        # if DEBUG_PRINT_RESPONSES:
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
