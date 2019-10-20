import {Action} from "redux";
import * as Effects from "redux-saga/effects";

export interface AppState {
    text: string,
    translatedText?: string
}

export interface PayloadAction<Type, Payload> extends Action<Type> {
    payload: Payload
}

export const AppActionTypes = {
    SET_INPUT_TEXT: "SET_INPUT_TEXT",
    SET_TRANSLATED_TEXT: "SET_TRANSLATED_TEXT"
};

export const AppActions = {
    setInputText: (text: string): PayloadAction<string, string> => {return {type: AppActionTypes.SET_INPUT_TEXT, payload: text}},
    setTranslatedText: (text: string): PayloadAction<string, string> => {return {type: AppActionTypes.SET_TRANSLATED_TEXT, payload: text}}
};

export function rootReducer(state: AppState = {text: "", translatedText: undefined}, action: PayloadAction<string, string>): AppState {
    switch (action.type) {
        case AppActionTypes.SET_INPUT_TEXT:
            return {
                text: action.payload,
                translatedText: "..."
            };
        case AppActionTypes.SET_TRANSLATED_TEXT:
            return {
                text: state.text,
                translatedText: action.payload
            };
        default:
            return state;
    }
}

async function translate(text: string): Promise<string> {
    try {
        let response = await fetch('/translate', {
            method: "POST",
            headers: new Headers({
                'Content-Type': 'application/json'
            }),
            body: JSON.stringify({text: text})
        });
        let body = await response.json();
        return body.text;
    } catch (error) {
        return "Error: " + error.toString()
    }
}

export function* translateSaga(action: PayloadAction<string, string>) {
    if (action.payload === "") {
        yield Effects.put(AppActions.setTranslatedText(""));
        return;
    }
    let translation = yield Effects.call(translate, action.payload);
    yield Effects.put(AppActions.setTranslatedText(translation));
}

export function* rootSaga() {
    yield Effects.all([
        Effects.debounce(500, AppActionTypes.SET_INPUT_TEXT, translateSaga)
    ]);
}