import React from 'react';
import ReactDOM from 'react-dom';
import createSagaMiddleware from 'redux-saga'
import './index.css';
import App from './App';
import {applyMiddleware, createStore} from "redux";
import {rootReducer, rootSaga} from "./translate";
import {Provider} from "react-redux";

let sagaMiddleware = createSagaMiddleware();
let store = createStore(rootReducer, applyMiddleware(sagaMiddleware));
sagaMiddleware.run(rootSaga);

ReactDOM.render(<Provider store={store}><App /></Provider> , document.getElementById('root'));

