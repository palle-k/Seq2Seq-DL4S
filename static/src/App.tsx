import React from 'react';
import './App.css';
import {AppActions, AppState, PayloadAction} from "./translate";
import {connect} from "react-redux";

interface AppProps {
    text?: string,
    translatedText?: string,
    setText?: (text: string) => void
}

class App extends React.Component<AppProps> {
    render(): any {
        return (
            <div className="App">
                <header className="App-header">
                    <h3>
                        Attention Seq2seq translator
                    </h3>
                </header>

                <div className={"TranslationContainer"}>
                    <div className={"TranslationSource TranslationBox"}>
                        <p className={"LanguageName"}>English</p>
                        <textarea
                            placeholder={"Enter Text..."}
                            className={"TranslationSourceText"}
                            value={this.props.text}
                            onChange={(event) => this.props.setText!(event.target.value)}
                        >
                        </textarea>
                    </div>
                    <hr />
                    <div className={"TranslationTarget TranslationBox"}>
                        <p className={"LanguageName"}>German</p>
                        <p className={this.props.translatedText!.length === 0 ? "TranslationTargetText empty" : "TranslationTargetText"}>
                            {this.props.translatedText!.length === 0 ? "Translation" : this.props.translatedText}
                        </p>
                    </div>
                </div>
            </div>
        );
    }
}

function mapStateToProps(state: AppState) {
    return {
        text: state.text,
        translatedText: state.translatedText !== undefined ? state.translatedText : ""
    }
}

function mapDispatchToProps(dispatch: (action: PayloadAction<string, string>) => void): AppProps {
    return {
        setText: text => dispatch(AppActions.setInputText(text))
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(App);
