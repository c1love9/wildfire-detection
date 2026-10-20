"""Shared monitoring-point editor and interactive map for live/history pages."""

from dataclasses import replace
import base64
import json

import pydeck as pdk
import streamlit as st

from alarm import LEVEL_NAMES
from monitoring import destination, load_monitoring_points, point_from_metadata, view_sector


def edit_monitoring_point(path, locked=False, point_id=None):
    with st.expander('监控点位置与视角（模拟）'):
        mode = st.selectbox('位置记录方式', ['模拟定位', '不记录位置'], disabled=locked)
        st.caption('固定坐标与手动角度用于课程演示，不读取电脑或摄像头的实际位置，也不控制云台。')
        if mode == '不记录位置':
            return None
        points = load_monitoring_points(path)
        if point_id is not None:
            points = [point for point in points if point.id == point_id]
            if not points:
                raise ValueError('通道关联的监控点不存在。')
        point = st.selectbox('模拟监控点', points, format_func=lambda item: item.name, disabled=locked)
        longitude_col, latitude_col = st.columns(2)
        longitude = longitude_col.number_input('模拟经度（WGS84）', min_value=-180.0, max_value=180.0,
                                              value=float(point.longitude), step=0.0001, format='%.6f', key=f'lon_{point.id}', disabled=locked)
        latitude = latitude_col.number_input('模拟纬度（WGS84）', min_value=-90.0, max_value=90.0,
                                            value=float(point.latitude), step=0.0001, format='%.6f', key=f'lat_{point.id}', disabled=locked)
        heading_col, pitch_col, fov_col = st.columns(3)
        heading = heading_col.number_input('模拟水平角（度）', min_value=0.0, max_value=360.0,
                                          value=float(point.heading), step=5.0, key=f'heading_{point.id}', disabled=locked)
        pitch = pitch_col.number_input('模拟俯仰角（度）', min_value=-90.0, max_value=90.0,
                                      value=float(point.pitch), step=5.0, key=f'pitch_{point.id}', disabled=locked)
        fov = fov_col.number_input('模拟水平视场角（度）', min_value=1.0, max_value=179.0,
                                  value=float(point.horizontal_fov), step=5.0, key=f'fov_{point.id}', disabled=locked)
        st.caption('水平角：0°向北、90°向东，顺时针；俯仰角：负值向下。检测时锁定，停止后可修改。')
        return replace(point, longitude=longitude, latitude=latitude, heading=heading, pitch=pitch, horizontal_fov=fov)


def location_deck(point, level=0, events=(), selected_id=None, basemap=True):
    """One selected point/sector; optional historical alarm-associated points."""
    marker = {'position': [point.longitude, point.latitude], 'name': point.name,
              'description': f'模拟监控点 · {LEVEL_NAMES[level]} · WGS84',
              'color': [220, 45, 45, 230] if level else [30, 110, 220, 230], 'radius': 10}
    history = []
    for event in events:
        location = point_from_metadata(event.get('metadata_json', '{}'))
        if location is not None:
            history.append({'position': [location.longitude, location.latitude],
                            'name': f"{location.name} · {event['id'][:12]}",
                            'description': f"模拟位置 · {LEVEL_NAMES[event['peak_level']]} · {event['started_at']} UTC",
                            'color': [220, 45, 45, 230] if event['id'] == selected_id else [230, 150, 40, 180],
                            'radius': 12 if event['id'] == selected_id else 7})
    layers = [
        pdk.Layer('PolygonLayer', id='view-sector', data=[{'polygon': view_sector(point)}],
                  get_polygon='polygon', get_fill_color=[30, 110, 220, 45], get_line_color=[30, 110, 220, 180],
                  stroked=True, line_width_min_pixels=1),
        pdk.Layer('PathLayer', id='view-direction', data=[{'path': [[point.longitude, point.latitude], destination(point, point.heading)]}],
                  get_path='path', get_color=[30, 110, 220], width_min_pixels=3),
        pdk.Layer('ScatterplotLayer', id='monitor-and-alarms', data=history or [marker],
                  get_position='position', get_fill_color='color', get_radius='radius', radius_units='pixels', pickable=True),
    ]
    # Streamlit 1.50 interprets None as an online basemap and its frontend
    # requires a style URL string. A data URL embeds a source-free style.
    empty_style = {'version': 8, 'sources': {}, 'layers': [
        {'id': 'background', 'type': 'background', 'paint': {'background-color': '#eaf0f6'}}]}
    empty_style_url = 'data:application/json;base64,' + base64.b64encode(json.dumps(empty_style).encode()).decode('ascii')
    return pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=point.latitude, longitude=point.longitude, zoom=17),
                    map_style=pdk.map_styles.CARTO_LIGHT if basemap else empty_style_url, map_provider='carto',
                    tooltip={'text': '{name}\n{description}'})


def show_location(point, level=0, events=(), selected_id=None, key='monitor_map'):
    if point is None:
        st.info('此观测未记录位置与视角；旧日志不会自动套用当前监控点。')
        return
    st.subheader('监控点位置与视角（模拟）')
    st.caption(f'{point.name} · 模拟坐标 WGS84 · 经度 {point.longitude:.6f}，纬度 {point.latitude:.6f}')
    heading_col, pitch_col, fov_col = st.columns(3)
    heading_col.metric('水平角', f'{point.heading:g}°')
    pitch_col.metric('俯仰角', f'{point.pitch:g}°')
    fov_col.metric('水平视场角', f'{point.horizontal_fov:g}°')
    basemap = st.checkbox('显示在线地图底图', value=True, key=f'{key}_basemap')
    st.pydeck_chart(location_deck(point, level, events, selected_id, basemap), height=340, key=key)
    st.caption('蓝色扇形仅示意水平视场方向（绘图半径100米），不代表覆盖距离或燃烧面积。底图需联网，可关闭底图保留点位与方向。')
    st.caption('红色表示报警关联的监控点；坐标属于摄像头监控点，不是图像中火焰的实测位置。')
