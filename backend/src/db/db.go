package src

import (
	"database/sql"
	"fmt"
	"log"

	_ "github.com/lib/pq"
)

const (
	host     = "localhost"
	port     = 5432
	user     = "postgres"
	password = ""
	dbname   = "postgres"
)

func checkError(err error) {
	if err != nil {
		log.Print("[DB]: ", err)
	}
}

func CreateAllTables(db *sql.DB) error {
	tables := []struct {
		name  string
		query string
	}{
		{
			name: "corporate_actions",
			query: `
				CREATE TABLE IF NOT EXISTS corporate_actions (
					id BIGSERIAL PRIMARY KEY,
					ticker VARCHAR(10) NOT NULL,
					action_date DATE NOT NULL,
					action_type VARCHAR(20) NOT NULL,
					adjustment_factor DECIMAL(10,6),
					dividend_amount DECIMAL(10,4),
					detection_method VARCHAR(20),
					verified BOOLEAN DEFAULT FALSE,
					created_at TIMESTAMPTZ DEFAULT NOW(),
					UNIQUE(ticker, action_date, action_type)
				)
			`,
		},
		{
			name: "market_data",
			query: `
				CREATE TABLE IF NOT EXISTS market_data (
					id BIGSERIAL PRIMARY KEY,
					ticker VARCHAR(10) NOT NULL,
					timestamp TIMESTAMPTZ NOT NULL,
					open DECIMAL(12,4) NOT NULL,
					high DECIMAL(12,4) NOT NULL,
					low DECIMAL(12,4) NOT NULL,
					close DECIMAL(12,4) NOT NULL,
					volume BIGINT NOT NULL,
					UNIQUE(ticker, timestamp)
				)
			`,
		},
	}

	for _, table := range tables {
		_, err := db.Exec(table.query)
		if err != nil {
			return fmt.Errorf("failed to create table %s: %w", table.name, err)
		}
		log.Printf("Table '%s' ensured", table.name)
	}

	return nil
}

func Db_connect() *sql.DB {
	// connection string
	psqlconn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
		host, port, user, password, dbname)

	// open database
	db, err := sql.Open("postgres", psqlconn)
	checkError(err)

	// check db connection FIRST
	err = db.Ping()
	checkError(err)

	fmt.Println("Connected!")

	// THEN create tables
	err = CreateAllTables(db)
	checkError(err)

	return db
}
