import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objs as go
import pandas as pd
import json

from hop.hexabundle_allocation.hector import constants as hector_constants

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

df_targets = pd.read_csv('/Users/samvaughan/Science/Hector/Targets/Commissioning/StarFields/results/StarCatalogues/Panstarrs_match_Gaia_120_m22_targets.csv')
df_standards = pd.read_csv('/Users/samvaughan/Science/Hector/Targets/Commissioning/StarFields/results/StarCatalogues/Panstarrs_match_Gaia_120_m22_standards.csv')
df_guides = pd.read_csv('/Users/samvaughan/Science/Hector/Targets/Commissioning/StarFields/results/StarCatalogues/Panstarrs_match_Gaia_120_m22_guides.csv')

df_targets['type'] = 0
df_standards['type'] = 1
df_guides['type'] = 2

df_targets['priority'] = 1
df_standards['priority'] = 1
df_guides['priority'] = 1

proximity_arcseconds = 220
proximity = proximity_arcseconds / 60 / 60
radius = 0.97 # Plate radius in degrees

x_centre = 120
y_centre = -22




template = 'plotly_white'
hoverlabel=dict(
    font_size=12)

opacity = 0.3
marker_sizes = {0:10, 1:8, 2:5}
colours = {0: 'lightskyblue', 1:'orange', 2:'green'}
symbols = {0: 'circle', 1:'square', 2:'diamond'}

def create_figure(x_centre=x_centre, y_centre=y_centre, size_in_pixels=700):
    # Default figure
    #f = px.scatter(df, x='RA', y='DEC', custom_data=['ID'], height=size_in_pixels, width=size_in_pixels)

    scatter_layout = go.Layout( 
        hoverlabel=hoverlabel, 
        template=template, 
        xaxis=dict(title="R.A.", showgrid=False), 
        yaxis=dict(title="DEC", scaleanchor='x', scaleratio=1, showgrid=False), 
        autosize=False, 
        width=size_in_pixels, 
        height=size_in_pixels, 
        margin={'autoexpand' : True, 't': 20, 'r': 150, 'b':0.0},
        legend = dict(orientation = 'v', yanchor = "top"),
        shapes=[{
                    'type': 'circle',
                    'x0': x_centre - radius,
                    'x1': x_centre + radius,
                    'xref': 'x',

                    'y0': y_centre - radius,
                    'y1': y_centre + radius,
                    'yref': 'y',
                    'layer': 'above'
                }],

            )


    trace_targets = go.Scatter(x=df_targets['RA'],
        y=df_targets['DEC'],
        customdata=df_targets.loc[:, ['ID', 'type']],
        mode='markers',
        name='Targets',
        marker=dict(
                    color=colours[0],
                    symbol=symbols[0],
                    size=marker_sizes[0],
                    opacity=opacity,
                    line=dict(
                        color='black',
                        width=2)
                    )
        )

    trace_guides = go.Scatter(x=df_guides['RA'],
        y=df_guides['DEC'],
        customdata=df_guides.loc[:, ['ID', 'type']],
        mode='markers',
        name='Guides',
        marker=dict(
                    color=colours[1],
                    symbol=symbols[1],
                    size=marker_sizes[1],
                    opacity=opacity,
                    line=dict(
                        color='black',
                        width=2)
                    )
        )

    trace_standards = go.Scatter(x=df_standards['RA'],
        y=df_standards['DEC'],
        customdata=df_standards.loc[:, ['ID', 'type']],
        mode='markers',
        name='Standard Stars',
        marker=dict(
                    color=colours[2],
                    symbol=symbols[2],
                    size=marker_sizes[2],
                    opacity=opacity,
                    line=dict(
                        color='black',
                        width=2)
                    )
        )

    trace_centre = go.Scatter(
        x=[x_centre],
        y=[y_centre],
        mode='markers',
        name="Tile Centre",
        hoverinfo='all',
        marker=dict(color='black', size=10, symbol='x')
            )

    fig = go.Figure(data=[trace_targets, trace_guides, trace_standards, trace_centre], layout=scatter_layout)

    # #fig.add_shape(type="circle",
    #     xref="x", yref="y",
    #     x0=x_centre - radius, y0=y_centre - radius,
    #     x1=x_centre + radius, y1=y_centre + radius,
    #     line_color="black",
    #     editable=True
    #     )

    return fig

f = create_figure()


# Guide, standard or normal hexabundle
hexabundle_selector_dropdown = dcc.Dropdown(
        id='hexabundle_dropdown',
        options=[
            {'label': 'Targets', 'value': 0},
            {'label': 'Guide Stars', 'value': 1},
            {'label': 'Standard Stars', 'value': 2}
        ],
        value=0)

# Now make a dictionary with the correct table corresponding to the correct integer
tables = {0:df_targets, 1:df_guides, 2:df_standards}

app.layout = html.Div([
                    html.Hr(),
                    html.H1("Hector Tiling App", style={'text-align':'center'}),
                    html.Div(children=[
                        html.Div(children=[
                        html.Div(["Tile RA: ", dcc.Input(id='input-RA', value=x_centre, type='number', step='any', debounce=True)]), 
                        html.Div(["Tile DEC: ", dcc.Input(id='input-DEC', value=y_centre, type='number', step='any', debounce=True)]),
        html.Hr(),
                        html.Button('Clear Selection', id='clear'),
                        html.Button("Download Tile File", id="download_tile_button"),
                        dcc.Download(id="download-tile-file"),
                        html.Button("Download Guide File", id="download_guide_button"),
                        dcc.Download(id="download-guide-file"),
        html.Hr(),
                        html.Div(['Select which objects to edit:', 
                            hexabundle_selector_dropdown
                            ], style={"width": "50%", 'display':'inline-block'},),
        html.Hr(),
                        html.Div(id='selected_targets', style={'display': 'none'}),
                        html.Div(id='targets_text_output'),
                        html.Div(id='selected_guides', style={'display': 'none'}),
                        html.Div(id='guides_text_output'),
                        html.Div(id='selected_standards', style={'display': 'none'}),
                        html.Div(id='standards_text_output')],
                            style={'display': 'inline-block', 'vertical-align': 'middle'}),
    #html.Hr(),         
                    dcc.Graph(id = 'scatter', figure=f,style={'display': 'inline-block', 'vertical-align': 'middle'})
                    ]),
    html.Hr(),
                    dcc.Slider(id='size', min=400, max=2000, step=50, value=900,
                        marks={x: str(x) for x in [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]})
])
                    #html.Div(id='selected_points'), #, style={'display': 'none'})),
                    #html.Div('deleted:'),
                    #html.Div(id='deleted_points') #, style={'display': 'none'}))



# @app.callback(Output('deleted_points', 'children'),
#             [Input('delete', 'n_clicks')],
#             [State('selected_points', 'children'),
#             State('deleted_points', 'children')])
# def delete_points(n_clicks, selected_points, delete_points):
#     print('n_clicks:',n_clicks)
#     if selected_points:
#         selected_points = json.loads(selected_points)
#     else:
#         selected_points = []

#     if delete_points:
#         deleted_points = json.loads(delete_points)
#     else:
#         deleted_points = []
#     ns = [p['pointNumber'] for p in selected_points]
#     new_indices = [df.index[n] for n in ns if df.index[n] not in deleted_points]
#     print('new',new_indices)
#     deleted_points.extend(new_indices)
#     return json.dumps(deleted_points)

def make_scatter_trace(points, colour, symbol, marker_size, df, text_label):

    trace_tile_members = go.Scatter(
        x=[p['x'] for p in points],
        y=[p['y'] for p in points],
        mode='markers',
        name=f"Tile Members: {text_label}",
        customdata=df.loc[:, ['ID', 'type']],
        hovertemplate =
        '<b>%{text}</b><br>' + 
        'RA = %{x}, Dec = %{y}',
        text=[f"{text_label}: {df.loc[df.index[p['pointIndex']], 'ID']}" for p in points],
        hoverinfo='all',
        marker=dict(color=colour, size=marker_size, symbol=symbol, showscale=False, line=dict(width=3, color='red'))
    )

    return trace_tile_members


@app.callback([Output('selected_targets', 'children'), 
                Output('selected_guides', 'children'), 
                Output('selected_standards', 'children')],
            [Input('scatter', 'clickData'),
                Input('clear', 'n_clicks'),
                Input('hexabundle_dropdown', 'value')],
            [State('selected_targets', 'children'), 
            State('selected_guides', 'children'), 
            State('selected_standards', 'children')])
def select_point(clickData, clear_clicked, hexabundle_dropdown, selected_targets, selected_guides, selected_standards):

    ctx = dash.callback_context
    ids = [c['prop_id'] for c in ctx.triggered]

    if selected_targets:
        selected_targets = json.loads(selected_targets)
    else:
        selected_targets = []

    if selected_guides:
        selected_guides = json.loads(selected_guides)
    else:
        selected_guides = []

    if selected_standards:
        selected_standards = json.loads(selected_standards)
    else:
        selected_standards = []

    # All three of our lists. This order must match the order of the dropdown box
    all_selections = [selected_targets, selected_guides, selected_standards]

    # Now make some quick lists of the IDs we've already selected
    already_selected_target_IDs = [p['customdata'][0] for p in selected_targets]
    already_selected_guide_IDs = [p['customdata'][0] for p in selected_guides]
    already_selected_standard_IDs = [p['customdata'][0] for p in selected_standards]

    already_selected_IDs = [already_selected_target_IDs, already_selected_guide_IDs, already_selected_standard_IDs]

    if 'scatter.clickData' in ids:
        if clickData:
            for p in clickData['points']:
                # If the point is in the correct layer (i.e. is a target if the targets box is selected, etc) and if it's not already been selected, add it to the list.
                if p['curveNumber'] == hexabundle_dropdown:
                    if p['customdata'][0] not in already_selected_IDs[hexabundle_dropdown]:
                        all_selections[hexabundle_dropdown].append(p)
                    else:
                        index = already_selected_IDs[hexabundle_dropdown].index(p['customdata'][0])
                        # Loop through the selections and remove one
                        del all_selections[hexabundle_dropdown][index]

    if 'clear.n_clicks' in ids:
        all_selections[hexabundle_dropdown] = []


    all_selections[0] = json.dumps(all_selections[0])
    all_selections[1] = json.dumps(all_selections[1])
    all_selections[2] = json.dumps(all_selections[2])
    return all_selections


# Update the Target text box
@app.callback(
    Output('targets_text_output', 'children'),
    Input('selected_targets', 'children')
)
def update_targets(selected_targets):

    if selected_targets:
        selected_targets = json.loads(selected_targets)
        IDs = [df_targets.loc[df_targets.index[p['pointIndex']], 'ID'].astype(str) for p in selected_targets]
        return f"Targets selected ({len(IDs)}/19)"#: {', '.join(IDs)}"
    else:
        return 'None'

# Update the Guides box
@app.callback(
    Output('guides_text_output', 'children'),
    Input('selected_guides', 'children')
)
def update_guides(selected_guides):

    if selected_guides:
        selected_guides = json.loads(selected_guides)
        IDs = [df_guides.loc[df_guides.index[p['pointIndex']], 'ID'].astype(str) for p in selected_guides]
        return f"Guides selected ({len(IDs)}/6)"#: {', '.join(IDs)}"
    else:
        return 'None'

# Update the text box
@app.callback(
    Output('standards_text_output', 'children'),
    Input('selected_standards', 'children')
)
def update_standards(selected_standards):

    if selected_standards:
        selected_standards = json.loads(selected_standards)
        IDs = [df_standards.loc[df_standards.index[p['pointIndex']], 'ID'].astype(str) for p in selected_standards]
        return f"Standards selected ({len(IDs)}/2)"#: {', '.join(IDs)}"
    else:
        return 'None'

@app.callback(Output('scatter', 'figure'),
            [Input('selected_targets', 'children'),
            Input('selected_guides', 'children'),
            Input('selected_standards', 'children'),
            Input('input-RA', 'value'),
            Input('input-DEC', 'value'),
                Input('size', 'value'),
                Input('hexabundle_dropdown', 'value')])
def scatter_plot(selected_targets, selected_guides, selected_standards, tile_RA, tile_DEC, size, hexabundle_dropdown):
    #global f
    #deleted_points = json.loads(deleted_points_state) if deleted_points_state else []
    f = create_figure(x_centre=tile_RA, y_centre=tile_DEC, size_in_pixels=size)
    #fig.update_layout(width=int(size), height=int(size))

    selected_targets = json.loads(selected_targets) if selected_targets else []
    selected_guides = json.loads(selected_guides) if selected_guides else []
    selected_standards = json.loads(selected_standards) if selected_standards else []
    
    if selected_targets:
        target_trace = make_scatter_trace(points=selected_targets, colour=colours[0], symbol=symbols[0], marker_size=marker_sizes[0] + 5, df=df_targets, text_label='Target')
        f.add_trace(target_trace)
    if selected_guides:
        target_trace = make_scatter_trace(points=selected_guides, colour=colours[1], symbol=symbols[1], marker_size=marker_sizes[1] + 5, df=df_guides, text_label='Guide')
        f.add_trace(target_trace)
    if selected_standards:
        target_trace = make_scatter_trace(selected_standards, colour=colours[2], symbol=symbols[2], marker_size=marker_sizes[2] + 5, df=df_standards, text_label='Standard')
        f.add_trace(target_trace)
        
    # Add little proximity circles to each target    
    for p in selected_targets + selected_guides + selected_standards:
            f.add_shape(type="circle",
            xref="x", yref="y",
            x0=p['x'] - proximity, y0=p['y'] - proximity, x1=p['x'] + proximity, y1=p['y'] + proximity,
            line=dict(color='black', width=1,
                                      dash='dash')
            )

    return f



def make_magnet_X_noDC(RA, tile_RA):

    return (RA - tile_RA) * hector_constants.HECTOR_plate_radius * 1e3

def make_magnet_Y_noDC(DEC, tile_DEC):

    return (DEC - tile_DEC) * hector_constants.HECTOR_plate_radius * 1e3


@app.callback(
    Output("download-tile-file", "data"),
    [Input("download_tile_button", "n_clicks"),
    Input('selected_targets', 'children'),
            Input('selected_standards', 'children'),
            Input('input-RA', 'value'),
            Input('input-DEC', 'value')],
    prevent_initial_call=True,
)
def download_tile_function(n_clicks, selected_targets, selected_standards, tile_RA, tile_DEC):

    ctx = dash.callback_context
    if ctx.triggered:
        if ctx.triggered[0]['prop_id'] == 'download_tile_button.n_clicks':

            selected_targets = json.loads(selected_targets) if selected_targets else []
            selected_standards = json.loads(selected_standards) if selected_standards else []

            # Make a combined data frame
            targets = df_targets.loc[[df_targets.index[p['pointIndex']] for p in selected_targets]]
            standards = df_standards.loc[[df_standards.index[p['pointIndex']] for p in selected_standards]]

            df = pd.concat((targets, standards))

            df['MagnetX_noDC'] = df.apply(lambda x: make_magnet_X_noDC(x['RA'], tile_RA), axis=1)
            df['MagnetY_noDC'] = df.apply(lambda x: make_magnet_Y_noDC(x['DEC'], tile_DEC), axis=1)

            df = df.loc[:, ['ID','RA','DEC','g_mag','r_mag','i_mag','z_mag','y_mag','GAIA_g_mag','GAIA_bp_mag','GAIA_rp_mag','Mstar','Re','z','GAL_MU_E_R','pmRA','pmDEC','priority','MagnetX_noDC','MagnetY_noDC','type']]
            
            file_for_download = dcc.send_data_frame(df.to_csv, "tile.csv", sep=',', index=False)

            # Add in our header
            file_for_download['content'] = f"# Target and Standard Star Star file from Sam's interactive app\n# {tile_RA} {tile_DEC}\n# Proximity Value: 220\n" + file_for_download['content']
            return file_for_download
        else:
            pass
    else:
        pass

@app.callback(
    Output("download-guide-file", "data"),
    [Input("download_guide_button", "n_clicks"),
            Input('selected_guides', 'children'), 
            Input('input-RA', 'value'),
            Input('input-DEC', 'value')],
    prevent_initial_call=True,
)
def download_guide_function(n_clicks, selected_guides, tile_RA, tile_DEC):

    ctx = dash.callback_context
    if ctx.triggered:
        if ctx.triggered[0]['prop_id'] == 'download_guide_button.n_clicks':
            selected_guides = json.loads(selected_guides) if selected_guides else []

            guides = df_guides.loc[[df_guides.index[p['pointIndex']] for p in selected_guides]]
        
            # Just select a subset of the columns       
            guides['MagnetX_noDC'] = guides.apply(lambda x: make_magnet_X_noDC(x['RA'], tile_RA), axis=1)
            guides['MagnetY_noDC'] = guides.apply(lambda x: make_magnet_Y_noDC(x['DEC'], tile_DEC), axis=1)

            guides = guides.loc[:, ['ID', 'RA', 'DEC', 'r_mag', 'type', 'pmRA', 'pmDEC', 'MagnetX_noDC', 'MagnetY_noDC']]
            file_for_download = dcc.send_data_frame(guides.to_csv, "guide_tile.csv", sep=' ', index=False)

            # Add in our header
            file_for_download['content'] = f"# Guide Star file from Sam's interactive app\n# {tile_RA} {tile_DEC}\n# Proximity Value: 220\n" + file_for_download['content']

            return file_for_download
    else:
        pass




if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port='8050')