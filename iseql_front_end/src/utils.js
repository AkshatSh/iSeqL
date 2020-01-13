
import Cookies from 'universal-cookie';
import {COOKIE_LOGIN} from './configuration';

export const COOKIES = new Cookies();

export function post_data(url = ``, data = {}) {
    // Default options are marked with *
    return fetch(url, {
        method: "POST", // *GET, POST, PUT, DELETE, etc.
        mode: "cors", // no-cors, cors, *same-origin
        cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
        credentials: "same-origin", // include, *same-origin, omit
        headers: {
            "Content-Type": "application/json",
            // "Content-Type": "application/x-www-form-urlencoded",
        },
        redirect: "follow", // manual, *follow, error
        referrer: "no-referrer", // no-referrer, *client
        body: JSON.stringify(data), // body data type must match "Content-Type" header
    })
    .then(response => response.json()); // parses response to JSON
}

export function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

export function getLast(array, val_default=null) {
    return get_index(array, array.length - 1, val_default);
}

export function get_index(array, index, val_default=null) {
    if (index >= 0 && index < array.length) {
        return array[index];
    }

    return val_default;
}

export function get_user_url(url, params={}) {
    const user_name = COOKIES.get(COOKIE_LOGIN);
    const qparams = [`user_name=${user_name}`];
    for (const paramKey in params) {
        const paramVal = params[paramKey];
        qparams.push(`${paramKey}=${paramVal}`);
    }
    return `${url}?${qparams.join('&')}`;
}

export function get_user() {
    const user_name = COOKIES.get(COOKIE_LOGIN);
    return user_name;
}

export function is_valid(object) {
    return object !== null && object !== undefined;
}

export function get_items(obj) {
    return Object.keys(obj).map(function(key) {
        return [(key), obj[key]];
    });
}

export function safe_fetch(url, error_callback, success_callback) {
    fetch(url).then(result => {
        if (!result.ok) {
            error_callback(result);
        }
        return result.json()
    }).then(response => {
        success_callback(response);
    });
}

export function set_array_props(array, props) {
    for (const i in array) {
        const item = array[i];
        array[i] = {
            ...item,
            ...props,
        };
    }
}

export function get_percentage(a, b) {
    if (b === 0) {
        // divide by 0
        return 0;
    }
    return a / b * 100;
}

export function key_by_string(o, s) {
    s = s.replace(/\[(\w+)\]/g, '.$1'); // convert indexes to properties
    s = s.replace(/^\./, '');           // strip a leading dot
    var a = s.split('.');
    for (var i = 0, n = a.length; i < n; ++i) {
        var k = a[i];
        if (k in o) {
            o = o[k];
        } else {
            return;
        }
    }
    return o;
}

export function prettyPrint(word) {
    if (word.length < 1) {
        return word;
    }
    const first = word[0].toUpperCase();
    const rest = word.slice(1).toLowerCase();
    return first + rest;
}

export default {};