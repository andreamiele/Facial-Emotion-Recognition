import axios from 'axios';
import React from 'react';

import {
    Button,
    Card, CardActions,
    CardContent,
    Grid,
    List, ListItem, ListItemIcon,
    Typography
} from '@material-ui/core';

import * as ReactDOM from "react-dom";
import {AttachFile} from "@material-ui/icons";

class LoginUI extends React.Component {

    state = {
        selectedFile: null,
        predictions: null
    };

    predictionText = () => {
        let typographies = [];
        let faceCount = 0;
        if (this.state.predictions != null) {
            for (const prediction of this.state.predictions) {
                faceCount++;
                typographies.push(<Typography variant="h5" key={faceCount}>
                    {"Face " + faceCount + ": " + prediction}
                </Typography>)
            }
        }
        return typographies;
    }

    // Updating the selectedFile when the user selects a file
    onFileChange = event => {
        this.setState({
            selectedFile: event.target.files[0],
            predictions: null
        })

        let reader = new FileReader();
        reader.onload = (e) => {
            this.setState({image: e.target.result});
        };
        reader.readAsDataURL(event.target.files[0]);
    };

    // Displaying file information when it is uploaded
    onFileUpload = () => {
        const formData = new FormData();

        // Update the formData object
        formData.append(
            "user_image",
            this.state.selectedFile,
            this.state.selectedFile.name
        );

        const headers = {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        };
        console.log(this.state.selectedFile);
        axios.post("http://localhost:8855/upload_file", formData, headers)
            .then(res => {
                console.log({res});
                console.log(res.data["emotions"])
                this.setState({
                    image: "data:image/png;base64," + res.data["image"],
                    predictions: res.data["emotions"]
                })
            })
            .catch(err => {
                console.error({err});
            });

    };

    fileData = () => {
        if (this.state.selectedFile) {
            return (
                <div>
                    <Typography variant="h4" align="left">
                        File Details
                    </Typography>
                    <Typography variant="body1">File Name: {this.state.selectedFile.name}</Typography>
                    <Typography variant="body1">File Type: {this.state.selectedFile.type}</Typography>
                    <Typography variant="body1">Last Modified:{" "}
                        {this.state.selectedFile.lastModifiedDate.toDateString()}
                    </Typography>
                </div>
            );
        }
    };

    constructor(props) {
        super(props);
        this.state = {
            isFetching: true,
            users: []
        };
    }

    render() {
        const style = {
            width: '100%',
            boxSizing: 'border-box',
            padding: '9px',
            resize: 'none',
            fontSize: '18px'
        };
        return (
            <React.Fragment>
                <Grid
                    container
                    direction="column"
                    justify="space-between"
                    alignItems="center"
                >
                    <Grid
                        container
                        item
                        justify="center"
                        alignItems="center"
                        xs={8}
                    >
                        <Card>
                            <CardContent>
                                <Typography variant="h3" align="center">
                                    Prediction from an Uploaded Image
                                </Typography>
                                <List
                                    component="nav">
                                    <ListItem>
                                        <ListItemIcon>
                                            <AttachFile/>
                                        </ListItemIcon>
                                        <Button
                                            component="label"
                                            size={"small"}
                                            color="primary"
                                        >
                                            Choose your Image
                                            <input type="file" onChange={this.onFileChange} hidden/>
                                        </Button>
                                    </ListItem>
                                    <ListItem>
                                        {this.fileData()}
                                    </ListItem>
                                </List>
                                <img id="target" src={this.state.image} width="100%"/>
                                {this.predictionText()}
                            </CardContent>
                            <CardActions>
                                <Button
                                    variant={"contained"}
                                    size={"small"}
                                    color="#9b59b6"
                                    onClick={this.onFileUpload}>
                                    Click to Run
                                </Button>
                            </CardActions>
                        </Card>
                    </Grid>
                </Grid>
            </React.Fragment>
        );
    }
}

const domContainer = document.querySelector('#login_container');
ReactDOM.render(<LoginUI/>, domContainer);